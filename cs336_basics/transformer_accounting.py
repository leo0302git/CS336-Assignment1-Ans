def task_a():
    '''Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?'''
    vocab_size = 50257  
    context_length =1024  
    num_layers = 48  
    d_model = 1600
    num_heads = 25  
    d_ff = 6400

    fp32_unit_mem = 4 # every single-precision floating point number requires 4 bytes
    trainable_param = 2 * vocab_size * d_model + num_layers * d_model * (4 * d_model + 3 * d_ff + 2) + d_model
    print('How many trainable parameters would our model have?')
    print('Answer: ', trainable_param)
    memory_needed = trainable_param * fp32_unit_mem
    print('Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?')
    print(f'Answer: {memory_needed} bytes, i.e., {memory_needed / (1024**3)} GB', )

def task_b(context_length:int=1024):
    '''Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.'''
    vocab_size = 50257  
    num_layers = 48  
    d_model = 1600
    num_heads = 25  
    d_ff = 6400
    seq_len = context_length
    B = 1
    QKV_count = 4 * 2 * B * seq_len * d_model * d_model * num_layers
    self_attn_count = 2 * 2 * B * seq_len * d_model * seq_len * num_layers
    SwigLU_count = 3 * 2 * B * seq_len * d_ff * d_model * num_layers
    Linear_count = 4 * B * seq_len * d_model * vocab_size
    count_list = {
        'QKV': QKV_count,
        'self_attn': self_attn_count,
        'Swiglu': SwigLU_count,
        'Linear': Linear_count
    }
    most_computational_expensive = max(count_list, key=count_list.get)
    total_FLOPs = 2 * B * seq_len * ( num_layers * (4 * d_model * d_model + 2 * d_model * seq_len + 3 * d_model * d_ff) + 2 * d_model * vocab_size )
    print(f'When batch_size B = {B}, the FLOPs count is {total_FLOPs}')
    print('That is, '"%.2e" % total_FLOPs, 'FLOPs\n')
    print(f'QKV_count percentage: {QKV_count / total_FLOPs * 100}%')
    print(f'self_attn_count percentage: {self_attn_count / total_FLOPs * 100}%')
    print(f'SwigLU_count percentage: {SwigLU_count / total_FLOPs * 100}%')
    print(f'Linear_count percentage: {Linear_count / total_FLOPs * 100}%')
    print('most_computational_expensive: ', most_computational_expensive)
def task_c():
    print('Based on your analysis above, which parts of the model require the most FLOPs?')
    print('The transformer blocks require the most FLOPs. To be more specific, it\'s the SwigLU computing part.')
def task_d():
    '''Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?'''
    vocab_size = 50257  
    context_length =1024  
    num_layers = 48  
    d_model = 1600
    num_heads = 25
    d_ff = 6400
    seq_len = context_length
    B = 1
    model_names = ['GPT-2 small', 'GPT-2 medium', 'GPT-2 large']
    model_param = [
        {'num_layers': 12, 'd_model': 768, 'num_heads': 12},
        {'num_layers': 24, 'd_model': 1024, 'num_heads': 16},
        {'num_layers': 36, 'd_model': 1280, 'num_heads': 20}
    ]
    for i in range(0, len(model_names)):
        num_layers = model_param[i]['num_layers']
        d_model = model_param[i]['d_model']
        QKV_count = 4 * 2 * B * seq_len * d_model * d_model * num_layers
        self_attn_count = 2 * 2 * B * seq_len * d_model * seq_len * num_layers
        SwigLU_count = 3 * 2 * B * seq_len * d_ff * d_model * num_layers
        Linear_count = 2 * B * seq_len * d_model * vocab_size
        count_list = {
            'QKV': QKV_count,
            'self_attn': self_attn_count,
            'Swiglu': SwigLU_count,
            'Linear': Linear_count
        }
        most_computational_expensive = max(count_list, key=count_list.get)
        total_FLOPs = 2 * B * seq_len * ( num_layers * (4 * d_model * d_model + 2 * d_model * seq_len + 3 * d_model * d_ff) + 2 * d_model * vocab_size )
        print('model size: ', model_names[i], ' total FLOPs: {:.2e}'.format(total_FLOPs))
        print(f'QKV_count percentage: {QKV_count / total_FLOPs * 100}%')
        print(f'self_attn_count percentage: {self_attn_count / total_FLOPs * 100}%')
        print(f'SwigLU_count percentage: {SwigLU_count / total_FLOPs * 100}%')
        print(f'Linear_count percentage: {Linear_count / total_FLOPs * 100}%')
        print('most_computational_expensive: ', most_computational_expensive)
        
    print('The transformer blocks always require the most FLOPs as the size increases')
def task_e():
    '''Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?'''
    task_b(context_length=16384)

if __name__ == '__main__':
    print("============Task (a)============")
    task_a()

    print("============Task (b)============")
    task_b()

    print("============Task (c)============")
    task_c()

    print("============Task (d)============")
    task_d()

    print("============Task (e)============")
    task_e()
