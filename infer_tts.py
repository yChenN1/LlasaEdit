from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
from datetime import datetime
from xcodec2.modeling_xcodec2 import XCodec2Model

llasa_1b ='HKUSTAudio/Llasa-1B'

tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
model = AutoModelForCausalLM.from_pretrained(llasa_1b)
model.eval() 
model.to('cuda')

model_path = "HKUSTAudio/xcodec2"  
 
Codec_model = XCodec2Model.from_pretrained(model_path)
Codec_model.eval().cuda()   

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
def gen_audio(data, save_path):
    for audio in tqdm(data):
        with torch.no_grad():
            input_text = audio['transcription']
            save_name = f"{Path(audio['src_path']).stem}_{audio['src_path'].split('/')[-2]}_to_{audio['trg_path'].split('/')[-2]}.wav"
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            # Tokenize the text
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
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
            gen_wav = Codec_model.decode_code(speech_tokens) 
        
        sf.write(f"{save_path}/{save_name}", gen_wav[0, 0, :].cpu().numpy(), 16000)




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


if __name__ == '__main__':
    # data_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/paired_emotion_with_instruct2.csv'
    # eval_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess/val_combined.csv'
    # reconstruct_eval_data(data_path, eval_path)
    eval_data_path = '/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/dataset/ESD_bin_reprocess/val_combined_eval.csv'
    eval_data = get_eval_data(eval_data_path)
    today = datetime.now().strftime("%Y%m%d")
    
    n_tts = True
    if n_tts:
        save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/esd_tts/{today}/temp_0.9"
        os.makedirs(save_path, exist_ok=True)
        gen_audio(eval_data, save_path)

    else:
        save_path = f"/mnt/fast/nobackup/scratch4weeks/yc01815/llasa/evaluation/esd_tts_w_prompt/{today}_2"
        os.makedirs(save_path, exist_ok=True)
        gen_audio_w_prompt(eval_data, save_path)