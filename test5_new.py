import pandas
from transformers import pipeline
from gensim.models import KeyedVectors
import pandas as pd
import sys
import os
import time
import traceback

start = time.time()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# D:/bert_test/dmis-lab/biobert-base-cased-v1.2
model = pipeline('fill-mask', model='D:/bert_test/dmis-lab/biobert-base-cased-v1.2', device=0)
print(model.device)
model2 = KeyedVectors.load_word2vec_format(os.path.join('D:/pycharm/pharse2vec'
                                                        '/pubmed_phrase_word_combo_skip_model_vectors_win_5.bin'),
                                           binary=True)
# 以 dataframe读入需要预测的文本（样本）
data2 = pd.read_table('./AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt', sep="|", header=None, encoding='cp1252'
                     )
# on_bad_lines='skip'
# data2的行数
eq = 0
uneq = 0
zero_mask = 0
key_error = 0
type_error = 0
runtime_error = 0
attribute_error = 0

num = len(data2)
data = pd.read_table('./Map.txt', sep="|")
ff = open('./res2_BIO.txt', 'w', encoding='utf-8')
ffu = open('./uneq_M.txt', 'w', encoding='utf-8')
ff_key = open('./key_error_M.txt', 'w', encoding='utf-8')
ff_type = open('./type_error_M.txt', 'w', encoding='utf-8')
ff_runtime = open('./runtime_error_M.txt', 'w', encoding='utf-8')
ff_attribute = open('./attribute_error_M.txt', 'w', encoding='utf-8')
ff_0mask = open('./0mask_M.txt', 'w', encoding='utf-8')

# with open('./with_number.txt', 'r', encoding='utf-8') as f:
# line = f.readlines()
for indexs in data2.index:
    try:
        SF = ""
        LF = ""
        M0 = []
        M1 = []
        M2 = []
        best = "[]"
        print(indexs)
        SF_intext = str(data2.loc[indexs][2])
        print(SF_intext)
        SF = str(data2.loc[indexs][0])
        print(SF)
        # newSF = " " + SF_intext
        # print(newSF)
        text = str(data2.loc[indexs][6])
        LF = str(data2.loc[indexs][1])
        newLF = "-".join(LF.split(" "))
        newLF = newLF.lower()
        li = len(SF_intext)
        print(li)
        l = len(SF)
        print(l)
        if li == l or li - l > 1:
            newSF = " " + SF_intext + " "
            print(newSF)
            text = text.replace(newSF, ' [MASK]' + " ")
        elif li - l == 1:
            newSF = " " + SF_intext
            print(newSF)
            sign = SF_intext[li - 1]
            print(sign)
            text = text.replace(newSF, ' [MASK]' + sign)
        print(text)

        # text = text.replace(SF, ' [MASK] ')
        # ff.write(text + "\n")
        count = text.count(' [MASK]')
        # SF = str(data2.loc[indexs][0])
        # LF = str(data2.loc[indexs][1])
        # newLF = "-".join(LF.split(" "))
        # text = str(data2.loc[indexs][6])
        # text = text.replace(SF, '[MASK]')
        # ff.write(text + "\n")
        # count = text.count('[MASK]')
        # print("num of SF:%d\t" % count)
        if count == 0:
            # print("No [MASK]")
            pre = " " + "\t"
            best = " " + "\t"
            ff.write(pre)
            ff.write(best+"\n")
            zero_mask += 1
            ff_0mask.write(str(indexs) + "|")
            ff_0mask.write(SF_intext + "|")
            ff_0mask.write(newLF + "|")
            ff_0mask.write(text + "\n")
            continue
        if count == 1:
            pred = model(text)
            # print(pred)
            pre_str0 = str(pred[0]['token_str']).lower()
            pre_str1 = str(pred[1]['token_str']).lower()
            pre_str2 = str(pred[2]['token_str']).lower()
            pre = pre_str0 + "|" + pre_str1 + "|" + pre_str2 + "\t"
            # print(pre)
            # print("pre_str is:%s\t" % pre_str)
            # print("___________________")
        if count == 2:
            pred = model(text)
            # print(pred)
            pre_str0 = str([pred[0][0]['token_str'], pred[1][0]['token_str']][
                               pred[0][0]['score'] < pred[1][0]['score']]).lower()
            pre_str1 = str([pred[0][1]['token_str'], pred[1][1]['token_str']][
                               pred[0][1]['score'] < pred[1][1]['score']]).lower()
            pre_str2 = str([pred[0][2]['token_str'], pred[1][2]['token_str']][
                               pred[0][2]['score'] < pred[1][2]['score']]).lower()
            pre = pre_str0 + "|" + pre_str1 + "|" + pre_str2 + "\t"
        ff.write(pre)

        # line[indexs] = line[indexs].strip() + "|" + pre + "\n"
        # print(pre)
        key = SF
        choose_data = data[data['SF'] == key]
        # print(choose_data)
        # SF对应的LF
        target = choose_data["LF"]
        newtarget = ""
        # M0 = []
        # M1 = []
        # M2 = []
        # best = "[]"
        # 加连字符“-”
        for phrase in target:
            newtarget = "-".join(phrase.split(" "))
            newtarget = str(newtarget).lower()
            # print(newtarget)
            if newtarget in model2:
                if pre_str0 in model2:
                    sim0 = model2.similarity(pre_str0, newtarget)
                else:
                    sim0 = 0
                if pre_str1 in model2:
                    sim1 = model2.similarity(pre_str1, newtarget)
                else:
                    sim1 = 0
                if pre_str2 in model2:
                    sim2 = model2.similarity(pre_str2, newtarget)
                else:
                    sim2 = 0
            else:
                sim0 = 0
                sim1 = 0
                sim2 = 0

            # if newtarget or pre_str0 not in model2:
            # sim0 = 0
            # if newtarget or pre_str1 not in model2:
            # sim1 = 0
            # if newtarget or pre_str2 not in model2:
            # sim2 = 0
            # else:
            # sim0 = model2.similarity(pre_str0, newtarget)
            # sim1 = model2.similarity(pre_str1, newtarget)
            # sim2 = model2.similarity(pre_str2, newtarget)

            I0 = [pre_str0, newtarget, sim0]
            I1 = [pre_str1, newtarget, sim1]
            I2 = [pre_str2, newtarget, sim2]

            # print(I)
            M0.append(I0)
            M1.append(I1)
            M2.append(I2)
        # M转为dataframe格式
        frame0 = pd.DataFrame(M0, columns=['1', '2', '3'])
        frame_sort0 = frame0.sort_values(by='3', ascending=False)
        frame_sort0.reset_index(drop=True, inplace=True)
        # print("similarity:")
        # print(frame_sort)
        best0 = frame_sort0.loc[0][1] + "|" + frame_sort0.loc[1][1] + "|" + frame_sort0.loc[2][1] + "|"

        frame1 = pd.DataFrame(M1, columns=['1', '2', '3'])
        frame_sort1 = frame1.sort_values(by='3', ascending=False)
        frame_sort1.reset_index(drop=True, inplace=True)
        # print("similarity:")
        # print(frame_sort)
        best1 = frame_sort1.loc[0][1] + "|" + frame_sort1.loc[1][1] + "|" + frame_sort1.loc[2][1] + "|"

        frame2 = pd.DataFrame(M2, columns=['1', '2', '3'])
        frame_sort2 = frame2.sort_values(by='3', ascending=False)
        frame_sort2.reset_index(drop=True, inplace=True)
        # print("similarity:")
        # print(frame_sort)
        best2 = frame_sort2.loc[0][1] + "|" + frame_sort2.loc[1][1] + "|" + frame_sort2.loc[2][1]
        best = best0 + best1 + best2 + "\t"
        # print(best)

        # line[indexs] = line[indexs].strip() + "|" + best + "\n"
        ff.write(best+"\n")
        # print("LF is %s" % newLF)
        # print("that is\t" + best)
        if str(frame_sort0.loc[0][1]) == newLF or str(frame_sort0.loc[1][1]) == newLF or str(
                frame_sort0.loc[2][1]) == newLF or \
                str(frame_sort1.loc[0][1]) == newLF or str(frame_sort1.loc[1][1]) == newLF or str(
            frame_sort1.loc[2][1]) == newLF or \
                str(frame_sort2.loc[0][1]) == newLF or str(frame_sort2.loc[1][1]) == newLF or str(
            frame_sort2.loc[2][1]) == newLF:
            eq += 1
            # ff.write(str(indexs) + "|")
            # ff.write(SF_intext + "|")
            # ff.write(newLF + "|")
            # ff.write(text + "|")
            # ff.write(pre + "|")
            # ff.write(best + "\n")
        if str(frame_sort0.loc[0][1]) != newLF and str(frame_sort0.loc[1][1]) != newLF and str(
                frame_sort0.loc[2][1]) != newLF and \
                str(frame_sort1.loc[0][1]) != newLF and str(frame_sort1.loc[1][1]) != newLF and str(
            frame_sort1.loc[2][1]) != newLF and \
                str(frame_sort2.loc[0][1]) != newLF and str(frame_sort2.loc[1][1]) != newLF and str(
            frame_sort2.loc[2][1]) != newLF:
            uneq += 1
            # ffu.write(str(indexs) + "|")
            # ffu.write(SF_intext + "|")
            # ffu.write(newLF + "|")
            # ffu.write(text + "|")
            # ffu.write(pre + "|")
            # ffu.write(best + "\n")

        # ff.write(line[indexs])

        # print([pred[0][0]['token_str'], pred[0][0]['score']])
        # print([pred[1][0]['token_str'], pred[1][0]['score']])
        # print("pre_str is:%s\t" % pre_str)
        # print("____________________")

        # vec = model2[pre_str]
        # print(vec)
        # 以dataframe读入含有缩略词与全称的文本

        # print(data)
        # name = data['NSF'].value_counts()

        # print("[EQUAL]")

        # print("___________________________________________________________________________")

    except RuntimeError:
        runtime_error += 1
        ff.write(" " + "\t" + " "+"\n")
        # exstr = traceback.format_exc()
        # ff_runtime.write(text)
        # ff_runtime.write(exstr)
        # ff_runtime.write("____________________________________" + "\n")
        continue
    except AttributeError:
        attribute_error += 1
        ff.write(" " + "\t" + " " + "\n")
        # ff_attribute.write(str(indexs) + "\t")
        # ff_attribute.write(text)
        # exstr = traceback.format_exc()
        # ff_attribute.write(exstr)
        # ff_attribute.write("____________________________________" + "\n")
        continue
    except TypeError:
        type_error += 1
        ff.write(" " + "\t" + " " + "\n")
        # ff_type.write(str(indexs) + "\t")
        # ff_type.write(text)
        # exstr = traceback.format_exc()
        # ff_type.write(SF + "\n")
        # ff_type.write(exstr)
        # ff_type.write("____________________________________" + "\n")
        continue
    except KeyError:
        key_error += 1
        ff.write(pre + "\t" + " " + "\n")
        # exstr = traceback.format_exc()
        # ff_key.write(str(indexs) + "\t")
        # ff_key.write(text)
        # ff_key.write(SF + "\n")
        # ff_key.write(pre + "\n")
        # ff_key.write(LF + "\n")
        # ff_key.write(best + "\n")
        # ff_key.write(exstr)
        # ff_key.write("___________________________________" + "\n")
        continue

# print("eq is %d\t" % eq)
# print("error is %d\t" % error)
ff.write("sum:%d\n" % num)
ff.write("eq:%d\n" % eq)
ff.write("uneq:%d\n" % uneq)
ff.write("key_error:%d\n" % key_error)
ff.write("runtime_error:%d\n" % runtime_error)
ff.write("type_error:%d\n" % type_error)
ff.write("attribute_error:%d\n" % attribute_error)
ff.write("zero_mask:%d\n" % zero_mask)
# accuracy = round(eq/(num-error), 4)
# ff.write("accuracy:%f" % accuracy)
end = time.time()
# print("运行时间：{}".format(end - start))
