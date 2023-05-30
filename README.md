Arabic deep-learning based diacritization models ([Shakkala](https://github.com/Barqawiz/Shakkala), [Shakkelha](https://github.com/AliOsm/shakkelha)) ported to PyTorch.

```python
# %% load models
from models import Shakkala, Shakkelha

shakkala = Shakkala(sd_path='./data/shakkala_second_model6.pth')
shakkelha = Shakkelha(sd_path='./data/shakkelha_rnn_3_big_20.pth')

# %% unvocalized input
input_text = "اللغة العربية هي أكثر اللغات السامية تحدثا، وإحدى أكثر اللغات انتشارا في العالم، يتحدثها أكثر من 467 مليون نسمة"

# %% shakkala output
out_shakkala = shakkala.predict(input_text)
print(out_shakkala)
>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مَنْ 467 مُلْيُونُ نُسْمَةَ

# %% shakkelha output
out_shakkelha = shakkelha.predict(input_text)
print(out_shakkelha)
>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مِنْ 467 مَلْيُونٍ نَسَمَةً


```
