from neuralnet.map import TextProcess
import ctcdecode
import torch

textprocess = TextProcess()

# 0 - 91
labels = [' ', 'a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô', 'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y', 'à', 'á', 'ả', 'ã', 'ạ', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'ầ', 'ấ', 'ẩ', 'ẩ', 'ẫ', 'ậ', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', '_']

def DecodeGreedy(output, blank_label=91, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2).squeeze()
    # print(output)
    # print(output.shape)
    # print(arg_maxes)
    # print(arg_maxes.shape)
    decode = []
    for i, index in enumerate(arg_maxes):
        # print(i,index)
        if index != blank_label:
            if collapse_repeated and index == arg_maxes[i-1]:
                continue
            decode.append(index.item())
    return textprocess.int_to_text_sequence(decode)

class CTCBeamDecoder:

    def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
        print("loading beam search with lm...")
        self.decoder = ctcdecode.CTCBeamDecoder(
            labels,
            beam_width=beam_size, blank_id=labels.index('_'),
            model_path=kenlm_path)
        print("finished loading beam search")

    def __call__(self, output):
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])