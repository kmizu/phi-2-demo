import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''
Translate the following C program into Java:
```c
void swap(int* a, int* b) {
  int t = *a;
  *a = *b;
  *b = t;
}
int main(void) {
    int a = 100, b = 200;
    swap(&a, &b);
    return 0;
}
```
''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=400)
text = tokenizer.batch_decode(outputs)[0]
print(text)
