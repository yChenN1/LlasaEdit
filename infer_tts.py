import os
import sys
import glob
import torch
import jiwer
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
from datetime import datetime
from whisper.normalizers import EnglishTextNormalizer
sys.path.append('/mnt/fast/nobackup/users/jz01101/cy/LlasaEdit/xcodec2')
from vq_process import extract_vq_code, reconstruct_from_vq_code
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# Set CUDA-related environment variables
os.environ["CUDA_HOME"] = os.path.expanduser("~/cuda-12.6")
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Set GCC compiler paths from conda
conda_prefix = os.environ.get("CONDA_PREFIX", "/path/to/your/conda/env")  # fallback if not in conda env
os.environ["CC"] = f"{conda_prefix}/bin/x86_64-conda-linux-gnu-gcc"
os.environ["CXX"] = f"{conda_prefix}/bin/x86_64-conda-linux-gnu-g++"
os.environ["LD_LIBRARY_PATH"] = (
    f"{conda_prefix}/lib:"
    f"{conda_prefix}/x86_64-conda-linux-gnu/lib:"
    f"{conda_prefix}/lib/gcc/x86_64-conda-linux-gnu/12.4.0:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)


# llasa_1b ='/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/LLaSA_training/online/finetune/0715_Etts/checkpoint-10000'

llasa_1b = "HKUSTAudio/Llasa-1B"
tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
model = AutoModelForCausalLM.from_pretrained(llasa_1b)
model.eval() 
model.to('cuda')


def ids_to_speech_tokens(speech_ids):
 
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


#TTS start!
def gen_audio(data, save_path, mode='tts'):
    for idx in tqdm(range(len(data))):
        with torch.no_grad():
            input_text = data[idx]['text']
            save_name = f"{Path(data[idx]['audio_path']).stem}.wav"
            style_instruction = data[idx]['caption']
            
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            # Tokenize the text
            if mode == 'tts':
                chat = [
                    {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                    {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
                ]
            else:
                input_text = (
                    "Instruction: {instruction}\n\n"
                    "Text: {text}"
                ).format(instruction=style_instruction, text=formatted_text)

                chat = [
                    {"role": "system", "content": "You are a helpful speech assistant. Your job is to generate speech that matches the provided style instruction as closely as possible."},
                    {"role": "user", "content": "Convert the text to speech: " + input_text},
                    {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
                ]

            input_ids = tokenizer.apply_chat_template(
                chat, 
                tokenize=True, 
                return_tensors='pt', 
                continue_final_message=True
            )
            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            
            def test(input_ids):
                text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')
                text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')
                speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
                speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
                # tokenize the audio token, make it for test
                base_path = f'/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/Expresso_ears_dataset_valid'
                audio_path = f"{base_path}/{data[idx+1]['audio_path']}"
                wav, sr = librosa.load(audio_path, sr=16000)
                assert sr == 16000, "Only supports 16kHz audio"
                with torch.no_grad():
                    vq_code = extract_vq_code(wav)  # (1, 1, T_code)
                speech_ids = vq_code[0, 0].cpu().numpy() + 128256 + 8
                formatted_input = torch.from_numpy(np.array(
                    [speech_understanding_start_id] +
                    speech_ids.tolist() +
                    [speech_understanding_end_id],
                    dtype=np.int32
                ))
                # find the position of text understanding start
                text_start_pos = (input_ids == text_understanding_start_id).nonzero(as_tuple=True)[0][0]
                # add the text understanding end to the input
                input_ids = torch.cat([input_ids[:text_start_pos], formatted_input, input_ids[text_start_pos:]], dim=0)

                return input_ids

            input_ids = test(input_ids.squeeze(0))
            input_ids = input_ids.unsqueeze(0).to('cuda')

            # Generate the speech autoregressively
            outputs = model.generate(
                input_ids,
                max_length=2048,  # We trained our model with a max length of 2048
                eos_token_id= speech_end_id ,
                do_sample=True,    
                top_p=0.95,           #  Adjusts the diversity of generated content
                temperature=0.95,   #  Controls randomness in output
                repetition_penalty=1.2
            )
            # Extract the speech tokens
            generated_ids = outputs[0][input_ids.shape[1]:-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   
            # Convert  token <|s_23456|> to int 23456 
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
            # Decode the speech tokens to speech waveform
            gen_wav = reconstruct_from_vq_code(speech_tokens) 
        
        # __import__('ipdb').set_trace()
        sf.write(f"{save_path}/{save_name}", gen_wav, 16000)


def gen_audio_w_prompt(data, save_path):
    # 给一个source speech 及其对应的文本作为prompt
    #TTS start!
    prompt_dict = {
        '0011': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0011/Surprise/0011_001401.wav',
        '0012': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0012/Surprise/0012_001401.wav',
        '0013': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0013/Surprise/0013_001401.wav',
        '0014': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0014/Surprise/0014_001401.wav',
        '0015': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0015/Surprise/0015_001401.wav',
        '0016': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0016/Surprise/0016_001401.wav',
        '0017': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0017/Surprise/0017_001401.wav',
        '0018': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0018/Surprise/0018_001401.wav',
        '0019': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0019/Surprise/0019_001401.wav',
        '0020': '/mnt/fast/nobackup/scratch4weeks/yc01815/Emotion_Speech_Dataset/English/0020/Surprise/0020_001401.wav'
    }
    
    prompt_text ="The nine the eggs, I keep."
  
    with torch.no_grad():
        for audio in tqdm(data):
            target_text = audio['transcription']
            save_name = f"{Path(audio['src_path']).stem}_{audio['src_path'].split('/')[-2]}_to_{audio['trg_path'].split('/')[-2]}.wav"
            # Encode the prompt wav
            prompt_wav = prompt_dict[audio['src_path'].split('/')[-3]]
            prompt_wav, sr = sf.read(prompt_wav)   # you can find wav in Files
            prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)  

            vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
            # print("Prompt Vq Code Shape:", vq_code_prompt.shape )   

            vq_code_prompt = vq_code_prompt[0,0,:]
            # Convert int 12345 to token <|s_12345|>
            speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
            input_text = prompt_text + target_text
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            # Tokenize the text and the speech prefix
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
            ]

            input_ids = tokenizer.apply_chat_template(
                chat, 
                tokenize=True, 
                return_tensors='pt', 
                continue_final_message=True
            )
            input_ids = input_ids.to('cuda')
            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

            # Generate the speech autoregressively
            outputs = model.generate(
                input_ids,
                max_length=2048,  # We trained our model with a max length of 2048
                eos_token_id= speech_end_id ,
                do_sample=True,
                top_p=1,           
                temperature=0.8,
            )
            # Extract the speech tokens
            generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   
            # Convert  token <|s_23456|> to int 23456 
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
            # Decode the speech tokens to speech waveform
            gen_wav = Codec_model.decode_code(speech_tokens) 
            # if only need the generated part
            # gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

            sf.write(f"{save_path}/{save_name}", gen_wav[0, 0, :].cpu().numpy(), 16000)


def reconstruct_eval_data(data_path,eval_path):
    transcriptions = []
    data_df = pd.read_csv(data_path)
    eval_df = pd.read_csv(eval_path)
    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        src_path = row['source']
        match = data_df[data_df['src_path'] == src_path]
        if not match.empty:
            transcriptions.append(match.iloc[0]['transcription'])
    eval_df["transcription"] = transcriptions
    eval_df.to_csv(f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess/val_combined_eval.csv", index=False)


def get_eval_data(data_path):
    data_df = pd.read_csv(data_path)
    return data_df.to_dict(orient='records')


def asr(audio_gen_path, audio_file):
    def load_model(model_id: str, device: str):
        """加载语音识别模型"""
        torch_dtype = torch.float16 if "cuda" in device else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        return asr_pipeline, processor
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = "openai/whisper-large-v3"
    asr_pipeline, processor = load_model(model_id, device)
    normalizer = EnglishTextNormalizer()
    audio_list = glob.glob(f"{audio_gen_path}/**/*.wav", recursive=True)
    audio_df = pd.read_csv(audio_file)
    trans_gt, trans_from_audio, audio_name = [], [], []

    for data in tqdm(audio_list):
        # ex04-ex01_disgusted_004_channel1_segment_131.24_138.95_Expresso_ears_dataset2_to_environment
        file_stem = Path(data).stem
        match = audio_df[audio_df['audio_path'].str.contains(file_stem, na=False)]
        if not match.empty:
            transcription_gt = normalizer(match.iloc[0]['text'])
        try:
            transcription = asr_pipeline(data)['text'] # gt
            transcription = normalizer(transcription)

            trans_gt.append(transcription_gt)   
            trans_from_audio.append(transcription)   
            audio_name.append(file_stem)
        except:
            print(file_stem)
            # assert False
    
    # __import__('ipdb').set_trace()

    if trans_gt and trans_from_audio:
        wer = jiwer.wer(trans_gt, trans_from_audio)
        print(f"WER: {wer * 100:.2f} %")
    else:
        print("No valid audio-transcription pairs found.")

    df = pd.DataFrame({
    'audio_name': audio_name,
    'trans_gt': trans_gt,
    'trans_from_audio': trans_from_audio,
    'wer': wer
    })

    # 保存为 CSV 文件
    df.to_csv(f'{audio_gen_path}/transcription_results.csv', index=False)

if __name__ == '__main__':
    # data_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/paired_emotion_with_instruct2.csv'
    # eval_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess/val_combined.csv'
    # reconstruct_eval_data(data_path, eval_data_path)
    eval_data_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/Speech_gen_dataset/gen_speech_v1/valid_instruct_0_1000.csv'
    eval_data = pd.read_csv(eval_data_path).to_dict(orient='records')
    today = datetime.now().strftime("%Y%m%d")
    
    mode_list = ['etts', 'tts']
    mode = mode_list[1]
    save_path = f"/mnt/fast/nobackup/scratch4weeks/jz01101/llasa/evaluation/{today}/{mode}/{llasa_1b.split('-')[1]}_waudio_woorder"
    os.makedirs(save_path, exist_ok=True)

    print('save audio in: ', save_path)
    gen_audio(eval_data[:500], save_path, mode)
    # asr('/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/20250716/etts/10000', eval_data_path)


    assert False
    n_tts = True
    if n_tts:
        save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/esd_tts/{today}/temp_0.9"
        os.makedirs(save_path, exist_ok=True)
        gen_audio(eval_data, save_path)

    else:
        save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/esd_tts_w_prompt/{today}_2"
        os.makedirs(save_path, exist_ok=True)
        gen_audio_w_prompt(eval_data, save_path)