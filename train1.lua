require 'nn'
require 'nngraph'
require 'rnn'
require 'hdf5'
require 'optim'
require 'cunn';
require 'cudnn';
require 'cutorch';
cjson = require('cjson')






local file = hdf5.open('out1.h5', 'r')
local data = file:read('/'):all()
file:close()


local file1 = io.open('vocab.json', 'r')
local text = file1:read()
file1:close()
vocab = cjson.decode(text)
--print(vocab_quest['test_questions'][1][1])






batch_size = 2
caption_input = 4500
embedding_size = 100
vocab_size = 628
hidden_state_size = 200
epochs = 50
time_steps = 10+1
data_size = 500
gd_params = {learningRate = 0.005}


model1 = nn.Sequential()
model1:add(nn.Linear(caption_input,embedding_size))

model2 = nn.Sequential()
model2:add(nn.LookupTable(vocab_size,embedding_size))


LSTM = nn.Sequential()
LSTM:add(nn.Sequencer(nn.LSTM(embedding_size,hidden_state_size)))
LSTM:add(nn.Sequencer(nn.Linear(hidden_state_size,vocab_size)))
LSTM:add(nn.LogSoftMax())


criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

w1,g1 = model1:getParameters()
w2,g2 = model2:getParameters()
w3,g3 = LSTM:getParameters()

num_iterations = math.floor(500/batch_size)




	gen_model = nn.Sequential()
	gen_model:add(LSTM:get(1))
	gen_model:add(LSTM:get(2))
	gen_model:add(nn.SoftMax())

-- function sample(index)
	
-- 	input1 = torch.Tensor(1,caption_input)
-- 	--input2 = torch.Tensor(1,10)
-- 	--input2[1] = data['test_questions'][index]
-- 	input1[1] = data['test_caption_embeddings'][index]
-- 	output1 = model1:forward(input1)
-- 	--output2 = model2:forward(input2)
-- 	--lstm_input = torch.cat(output1:resize(1,embedding_size),output2:resize(time_steps-1,embedding_size),1)
-- 	generated = ""	

-- 	curr = gen_model:forward(output1)
-- 	curr = torch.multinomial(curr,1)
-- 	--curr = torch.Tensor{1,1}
-- 	--curr[1][1] = indexes[1][1]
-- 	generated = generated..' '..vocab['ind2word'][tostring(curr[1][1])]

-- 	--print(lstm_input)

	
	

-- 		--print(indexes[1])
-- 	for i = 1,time_steps - 1 do
-- 		output = model2:forward(curr)
-- 		curr = gen_model:forward(output)
-- 		curr = torch.multinomial(curr,1)

-- 		generated = generated..' '..vocab['ind2word'][tostring(curr[1][1])]
-- 	end
-- 	print(generated)
	

-- end





for i = 1,epochs do
	shuffle = torch.randperm(data_size)
	loss_e = 0
	for j = 1,num_iterations do
		data1 = torch.Tensor(batch_size,caption_input)
		data2 = torch.Tensor(batch_size,time_steps-1)
		target = torch.Tensor(batch_size,time_steps)
		for k = 1,batch_size do
			data1[k] = data['train_caption_embeddings'][shuffle[k+(j-1)*batch_size]]:resize(1,caption_input)
			data2[k] = data['train_questions'][shuffle[k+(j-1)*batch_size]]:resize(1,time_steps-1)
			for l = 1,time_steps do
				if l == time_steps then
					target[k][l] = vocab['word2ind']['<END>']
				else
					target[k][l] = data2[k][l]
				end
			end		
		end
		--print(target)
		for_data1  = model1:forward(data1)
		for_data2  = model2:forward(data2)

		lstm_input = torch.Tensor(batch_size,time_steps,embedding_size)
		for t = 1,batch_size do
			lstm_input[t] = torch.cat(for_data1[t]:resize(1,embedding_size),for_data2[t]:resize(time_steps-1,embedding_size),1)
		end
		--print(lstm_input:size())
		lstm_output = LSTM:forward(lstm_input)

		-- if j == 2 and i == 1 then
		-- 	print(lstm_output)
		-- end	
		loss = criterion:forward(lstm_output,target)
		-- if j%10 == 0 then
		-- 	print(loss,i)
		-- end	
		loss_e = loss_e + loss
		dc = criterion:backward(lstm_output,target)
		model1:zeroGradParameters()
		model2:zeroGradParameters()
		LSTM:zeroGradParameters()

		dl = LSTM:backward(lstm_input,dc)
		g3:clamp(-5,5)
		dm1 = dl[{{},{1},{}}]:resize(batch_size,embedding_size)
		dm2 = dl[{{},{2,time_steps},{}}]
		
		model1:backward(data1,dm1)
		g1:clamp(-5,5)
		model2:backward(data2,dm2)
		g2:clamp(-5,5)


		local feval3 = function(w_new3)
       		collectgarbage()

            -- reset data
            if w3 ~= w_new3 then w3:copy(w_new3) end 
    		return loss, g3
    	end

    	local feval1 = function(w_new1)
       		collectgarbage()

            -- reset data
            if w1 ~= w_new1 then w1:copy(w_new1) end 
    		return loss, g1
    	end	

    	local feval2 = function(w_new2)
       		collectgarbage()

            -- reset data
            if w2 ~= w_new2 then w2:copy(w_new2) end 
    		return loss, g2
    	end
    		
    	r3,t3 = optim.adam(feval3,w3,gd_params)
		r1,t1 = optim.sgd(feval1,w1,gd_params)
		r2,t2 = optim.sgd(feval2,w2,gd_params)

		-- if j == 1 then
		-- 	print(data1,data2)
		-- end
	end
	loss_e = loss_e/data_size
	--sample(index)
	print(loss_e,i)
end	

-- --print(type(vocab['word2ind']['man']))

-- --sample(1)

	
	
