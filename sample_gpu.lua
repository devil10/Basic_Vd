require 'nn'
require 'nngraph'
require 'rnn'
require 'hdf5'
require 'optim'
require 'cunn';
require 'cudnn';
require 'cutorch';
cjson = require('cjson')

-- local file = hdf5.open('../visdial/data/data_img.h5', 'r')
-- local data_1 = file:read('/'):all()
-- file:close()


local file = hdf5.open('out1.h5', 'r')
local data = file:read('/'):all()
file:close()


local file1 = io.open('vocab.json', 'r')
local text = file1:read()
file1:close()
vocab = cjson.decode(text)


model1 = torch.load('models/model_1_epoch10')
model2 = torch.load('models/model_2_epoch10')
model = torch.load('models/model_epoch10')


caption_input = 4500
vocab_size = 628

function sample(index1)
	
	input1 = torch.Tensor(1,caption_input)
	input1[1] = data['test_caption_embeddings'][index1]
	input1 = input1:cuda()
	output1 = model1:forward(input1)
	generated = ""

	gen_model = nn.Sequential()
	gen_model:add(model:get(1))
	gen_model:add(model:get(2))
	gen_model:add(nn.SoftMax())

	generator = nn.Sequencer(gen_model)
	generator  = generator:cuda()

	curr = generator:forward(output1)
	--print(curr)
	curr = torch.multinomial(curr,1)
	generated = generated..' '..vocab['ind2word'][tostring(curr[1][1])]

	--print(lstm_input)

	--print(curr)
	

		--print(indexes[1])
	while curr[1][1] ~= vocab['word2ind']['<END>'] do
		output = model2:forward(curr)
		--print(output)
		curr = generator:forward(output)
		curr =curr:resize(1,vocab_size)
		curr = torch.multinomial(curr,1)

		generated = generated..' '..vocab['ind2word'][tostring(curr[1][1])]
		-- curr = curr:cuda()
		-- b = curr
		-- b = b:cuda()
	end
	print(generated)
	

end


print(sample(1))