# encoding = utf8


def change_to_bio():
    train_file = 'data/train.txt'
    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('  ')
            sentence = []
            for token in tokens:
                words, tag= token.split('/')
                words = words.split('_')
                if tag == 'o':
                    for word in words:
                        sentence.append(word+' '+tag)
                else:
                    sentence.append(words[0]+' B-'+tag)
                    for word in words[1:]:
                        sentence.append(word + ' I-' + tag)
            data.append(sentence)
    return data


def create_vocab():
    dico = {}
    with open('data/corpus.txt', 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split('_')
            for item in words:
                if item not in dico:
                    dico[item] = 1
                else:
                    dico[item] += 1

    dico['<S>'] = 1000000002
    dico['</S>'] = 1000000001
    dico['<UNK>'] = 1000000000
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    print(sorted_items)

    with open('data/vocab.txt', 'w', encoding='utf-8') as f:
        for item, count in sorted_items:
            f.write(item)
            f.write('\n')


def ner_data_stat():
    cnt_test = 0
    with open('data/test.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip().split('_')
            if len(data) > 126:
                cnt_test += 1
    print("有{}条测试数据长度大于126。".format(cnt_test))

    cnt_train = 0
    with open('data/train.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('  ')
            sentence = []
            for token in tokens:
                words, tag = token.split('/')
                words = words.split('_')
                if tag == 'o':
                    for word in words:
                        sentence.append([word, 'O'])
                else:
                    sentence.append([words[0], 'B-' + tag])
                    for word in words[1:]:
                        sentence.append([word, 'I-' + tag])
            if len(sentence) > 126:
                cnt_train += 1
    print("有{}条训练数据长度大于126。".format(cnt_train))


def gaopincidebiaoji():
    gaopinci = []
    with open('data/vocab.txt', 'r', encoding='utf-8') as f:
        for i in range(10):
            line = f.readline().strip()
            gaopinci.append(line)
    gaopinci = gaopinci[3:]

    confirmed_not_tag_word = gaopinci[2]    # 15274
    # print(confirmed_not_tag_word)
    one_not_tag_word = gaopinci[0]  # 只出现在了一句长度小于126的句子中，标记为a。可能是"的"一类的字
    # print(one_not_tag_word)     # 21224

    # most_word = gaopinci[6]
    train, test = get_ner_long_sent()

    shorter = []
    print("共有{}条NER训练数据的长度大于126".format(len(train)))

    # 拿掉长度大于252的句子
    len_126_252 = []
    len_252 = []
    for item in train:
        sent, tags = item
        if len(sent) < 252:
            len_126_252.append(item)
        else:
            len_252.append(item)
    print("共有{}条NER训练数据的长度介于126到252之间".format(len(len_126_252)))

    train_dict = []
    success = 0
    duandian = []
    for sent, tags in len_126_252:
        index = []
        for idx, word in enumerate(sent):
            if word == one_not_tag_word or word == confirmed_not_tag_word:
                index.append(idx)
                if tags[idx] != 'O':
                    print("error")

        split = int(len(sent)/2)
        for i in range(len(index) - 1):
            if abs(index[i] - split) <= abs(index[i+1] - split):
                split = index[i]
                break

        if split < 126 and (len(sent) - split) < 126:
            success += 1
        else:
            train_dict.append({'res': len(sent)-split,
                               'indexes': index,
                               'split': split})
        if not index:
            print(len(sent))
            print(' '.join(sent))
        duandian.append(index)
    print(success)

    # print("共有{}条NER测试数据的长度大于126".format(len(test)))
    # cnt = 0
    # for sent in test:
    #     if confirmed_not_tag_word in sent:
    #         cnt += 1
    #     else:
    #         if one_not_tag_word in sent:
    #             cnt += 1
    # print(cnt)



def get_ner_long_sent():
    # get sentences longer than 126 in ner data
    train_sents = []
    with open('data/train.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('  ')
            sentence = []
            tags = []
            for token in tokens:
                words, tag = token.split('/')
                words = words.split('_')
                if tag == 'o':
                    for word in words:
                        sentence.append(word)
                        tags.append('O')
                else:
                    for word in words:
                        sentence.append(word)
                    tags.append('B-' + tag)
                    for word in words[1:]:
                        tags.append('I-' + tag)
            if len(sentence) > 126:
                train_sents.append([sentence, tags])

    test_sents = []
    with open('data/test.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip().split('_')
            if len(data) > 126:
                test_sents.append(data)
    return train_sents, test_sents

if __name__ == '__main__':
    #sentences = change_to_bio()

    #print("共有:{}条训练集中句子长度超过512".format(count))
    #print("训练集中最长的句子长度为：{}".format(max_len))

    gaopincidebiaoji()