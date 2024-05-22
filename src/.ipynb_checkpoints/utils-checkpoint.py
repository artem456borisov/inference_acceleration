
choices = ["A", "B", "C", "D"]

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))



def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def format_example_rummlu(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, 1])
    return prompt



def gen_prompt(train_df, subject, data_dir, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        if data_dir == 'data/':
            prompt += format_example(train_df, i)
        else:
            prompt += format_example_rummlu(train_df, i)
    return prompt



def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(DEVICE)

    return input_tokens

#
#used specifically for llama.cpp
def generate_text_from_prompt(user_prompt,
                              model,
                             max_tokens = 1,
                             temperature = 0,
                             echo = True):
    model_output = model(
       user_prompt,
       max_tokens=max_tokens,
       temperature=temperature,
       echo=echo
    )
    return model_output['choices'][0]['text']


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts, batch_size):
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))