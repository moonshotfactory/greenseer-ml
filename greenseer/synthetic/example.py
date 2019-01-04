#%%
from gluoncv.data import RecordFileDetection
record_dataset = RecordFileDetection('val.rec', coord_normalized=True)

#%%
len(record_dataset)

#%%
type(record_dataset[0]), type(record_dataset[1])
len(record_dataset[0])

#%%
