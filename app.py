import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# model
def generate_description(notes):
    prompt = f"Write a product description based on the below information.\n\n{notes}\n\nDescription:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=256, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description

# Entry-point with Streamlit code
def main():
    st.title("Product Description Generator")
    notes = st.text_area("Enter product information:")
    if st.button("Generate description"):
        with st.spinner("Generating description..."):
            description = generate_description(notes)
        st.subheader("Generated description:")
        st.write(description)

if __name__=="__main__":
    main()