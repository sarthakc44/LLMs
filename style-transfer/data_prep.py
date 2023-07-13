import os
import pandas as pd

#############################################################################################################
#meeting conversation dialogues

meeting_conv = {}
for dirname, _, filenames in os.walk('data\conversation meetings'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        #print(path, filename, filename[-5])
        with open(path, 'r') as file:
            meeting_conv[filename[-5]] = file.read()
            #print(filename)


###############################################################################################################
"""
star trek dialogues are by character, not scene
house md dialogues have no scene/episode marking

to check:
NPR interview dialogues
bbt
avatar
"""


###############################################################################################################
#democratic debate

debate_df = pd.read_csv("data\\USdemocraticdebates2020\\debate_transcripts.csv")
cols = ['date', 'debate_name', 'debate_section']
debate_df['combined'] = debate_df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1) #combine 3 index columns to make unique key for filtering

debates_list = [debate_df[i].unique().tolist() for i in debate_df[['combined']].columns] #list of 1 list
debates_list = [item for sublist in debates_list for item in sublist] # flatten list

cols = ['speaker','speech']
debate_df['dialogue'] = debate_df[cols].apply(lambda row: ': '.join(row.values.astype(str)), axis=1) #combine speaker with dialogue

df_dialogues = debate_df[['combined','dialogue']]

debate_txt = {}
for idx in range(0, len(debates_list)):
    temp_df = df_dialogues.loc[df_dialogues['combined'] == debates_list[idx]].drop(columns = 'combined')
    temp_list = temp_df['dialogue'].values.tolist()

    temp_str = debates_list[idx] + '\n\n'
    for item in temp_list:
        temp_str += item + '\n\n'
    debate_txt[idx] = temp_str    # final dictionary of debates

#print(debate_txt[1])

"""

for idx in range(0, len(debates_list)):
    temp_list = debate_txt[idx]
    name = 'debate' + str(idx) + '.txt'
    path = 'data\\USdemocraticdebates2020\\' + name
    with open(path, "w") as f:
        f.write(temp_list)

"""