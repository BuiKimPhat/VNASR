import torch

class TextProcess:
	def __init__(self):
		char_map_str = """
		<SPACE> 0
		a 1
		ă 2
		â 3
		b 4
		c 5
		d 6
		đ 7
		e 8
		ê 9
		g 10
		h 11
		i 12
		k 13
		l 14
		m 15
		n 16
		o 17
		ô 18
		ơ 19
		p 20
		q 21
		r 22
		s 23
		t 24
		u 25
		ư 26
		v 27
        x 28
        y 29
		à 30
		á 31
		ả 32
		ã 33
		ạ 34
		ằ 35
		ắ 36
		ẳ 37
		ẵ 38
		ặ 39
		ầ 40
		ấ 41
		ẩ 42
		ẩ 43
		ẫ 44
		ậ 45
		è 46
		é 47
		ẻ 48
		ẽ 49
		ẹ 50
		ề 51
		ế 52
		ể 53
		ễ 54
		ệ 55
		ì 56
		í 57
		ỉ 58
		ĩ 59
		ị 60
		ò 61
		ó 62
		ỏ 63
		õ 64
		ọ 65
		ồ 66
		ố 67
		ổ 68
		ỗ 69
		ộ 70
		ờ 71
		ớ 72
		ở 73
		ỡ 74
		ợ 75
		ù 76
		ú 77
		ủ 78
		ũ 79
		ụ 80
		ừ 81
		ứ 82
		ử 83
		ữ 84
		ự 85
		ỳ 86
		ý 87
		ỷ 88
		ỹ 89
		ỵ 90
		"""
		self.char_map = {}
		self.index_map = {}
		for line in char_map_str.strip().split('\n'):
			ch, index = line.split()
			self.char_map[ch] = int(index)
			self.index_map[int(index)] = ch
		self.index_map[0] = ' '

	def text_to_int_sequence(self, text):
		""" Use char map to convert text to an integer sequence """
		text = text.lower()
		int_sequence = []
		for c in text:
			if c == ' ':
				ch = self.char_map['<SPACE>']
			else:
				ch = self.char_map[c]
			int_sequence.append(ch)
		return int_sequence

	def int_to_text_sequence(self, labels):
		""" Use index map to convert integer labels to a text sequence """
		string = []
		for i in labels:
			string.append(self.index_map[i])
		return ''.join(string).replace('<SPACE>', ' ')

# 0 - 91
labels = [' ', 'a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô', 'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y', 'à', 'á', 'ả', 'ã', 'ạ', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'ầ', 'ấ', 'ẩ', 'ẩ', 'ẫ', 'ậ', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', "_"]
def DecodeGreedy(output, blank_label=91, collapse_repeated=True):
    tp = TextProcess()
    arg_maxes = torch.argmax(output, dim=2).squeeze(0)
    decode = []
    for i, index in enumerate(arg_maxes):
        if index != blank_label:
            if collapse_repeated and index == arg_maxes[i-1]:
                continue
            decode.append(index.item())
    return tp.int_to_text_sequence(decode)