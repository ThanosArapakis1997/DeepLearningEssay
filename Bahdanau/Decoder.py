import torch
import torch.nn as nn
import Attention
import torch.nn.functional as F

MAX_LENGTH= 10
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):        
        super(Decoder, self).__init__()    
        
        self.output_dim = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions
        # input = [batch_size]
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [src_len, batch_size, enc_hid_dim * 2]
        
        input = input.unsqueeze(0)
        # input = [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        # Compute attention weights
        attn_weights = self.attention(hidden, encoder_outputs)
        # attn_weights = [batch_size, src_len]
        
        attn_weights = attn_weights.unsqueeze(1)
        # attn_weights = [batch_size, 1, src_len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch_size, src_len, enc_hid_dim * 2]
        
        context = torch.bmm(attn_weights, encoder_outputs)
        # context = [batch_size, 1, enc_hid_dim * 2]
        
        context = context.permute(1, 0, 2)
        # context = [1, batch_size, enc_hid_dim * 2]
        
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [1, batch_size, dec_hid_dim]
        # hidden = [1, batch_size, dec_hid_dim]
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context = context.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        # prediction = [batch_size, output_dim]
        
        return prediction, hidden.squeeze(0)