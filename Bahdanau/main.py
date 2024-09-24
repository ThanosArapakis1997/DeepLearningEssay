import torch 
import random

from BahdanauApproach import train
from data_preparation import get_dataloader
from Encoder import Encoder
from Decoder import Decoder
from data_preparation import tensorFromSentence

SOS_token = 0
EOS_token = 1



def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
    
    
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


#main 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size)

encoder = Encoder(input_lang.n_words, hidden_size)#.to(device)
decoder = Decoder(hidden_size, output_lang.n_words)#.to(device)

train(train_dataloader, encoder, decoder, 2, print_every=5, plot_every=5)

encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)