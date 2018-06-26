require 'nn'
require 'nngraph'
require 'rnn'
require 'hdf5'
require 'optim'
require 'cunn';
require 'cudnn';
require 'cutorch';
cjson = require('cjson')




local file = hdf5.open('../visdial/data/data_img.h5', 'r')
local data_1 = file:read('/'):all()
file:close()


--data_1['images_test'] = data_1['images_train'][{{5001,6000},{}}]

local file = hdf5.open('out1.h5', 'r')
local data = file:read('/'):all()
file:close()


local file1 = io.open('vocab.json', 'r')
local text = file1:read()
file1:close()
vocab = cjson.decode(text)
--print(vocab['ind2word'])


batch_size = 10
caption_input = 4096
embedding_size = 100
vocab_size = 629
hidden_state_size = 200
epochs = 50
ques_len = 10
time_steps = 10+1
data_size = 3363
gd_params = {learningRate = 0.005}

--print(data['train_caption_embeddings'][1])
print('heyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
model1 = nn.Sequential()
model1:add(nn.Linear(caption_input,embedding_size))
model1 = model1:cuda()

--data['train_caption_embeddings'][{{3363},{}}] = data['train_caption_embeddings'][{{3362},{}}]
--data['train_questions'][{{3363},{}}] = data['train_questions'][{{3362},{}}]
--data['train_questions'][{{4012},{}}] = data['train_questions'][{{3362},{}}]


model2 = nn.Sequential()
model2:add(nn.LookupTableMaskZero(vocab_size,embedding_size))
model2 = model2:cuda()



a = torch.CudaTensor{1,628,0}:resize(1,3)
print(model2:forward(a))
-- LSTM = nn.Sequential()
-- LSTM:add(nn.Sequencer(nn.LSTM(embedding_size,hidden_state_size)))
-- LSTM:add(nn.Sequencer(nn.Linear(hidden_state_size,vocab_size)))
-- LSTM:add(nn.LogSoftMax())

model = nn.Sequential()
model:add(nn.MaskZero(nn.LSTM(embedding_size,hidden_state_size),1))
model:add(nn.MaskZero(nn.Linear(hidden_state_size,vocab_size),1))
model:add(nn.MaskZero(nn.LogSoftMax(),1))

LSTM = nn.Sequencer(model)
LSTM = LSTM:cuda()



criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
criterion = criterion:cuda()



w1,g1 = model1:getParameters()
w2,g2 = model2:getParameters()
w3,g3 = LSTM:getParameters()

losss = -1*torch.log(1.0/vocab_size)*time_steps
num_iterations = math.floor(data_size/batch_size)
print(num_iterations)


--print(data['train_questions'][{{1},{2,4}}])
--print(vocab['word2ind']['<START>'])


function length(x)
	local count = 0
	for i = 1,ques_len do
		if x[i] ~=  0 then
			count = count +1
		end
	end
	return count
end	

--print(length(data['train_questions'][1]))			


	-- gen_model = nn.Sequential()
	-- gen_model:add(model:get(1))
	-- gen_model:add(model:get(2))
	-- gen_model:add(nn.SoftMax())

	-- generator = nn.Sequencer(gen_model)


function sample(index1)
	
	input1 = torch.Tensor(1,caption_input)
	input1[1] = data_1['images_train'][index1]
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



-- function dataloader(iter_no)
-- 	data1 = torch.Tensor(batch_size,caption_input)
-- 	data2 = torch.Tensor(batch_size,time_steps)
-- 	target

index = 1

--print(torch.CudaTensor(1,3))

for i = 1,epochs do
	--shuffle = torch.randperm(data_size)
	shuffle = torch.range(1,data_size)
	loss_e = 0
	for j = 1,num_iterations do
		--collectgarbage()
		print(i,j)
		data1 = torch.zeros(batch_size,caption_input)
		data2 = torch.zeros(batch_size,ques_len+1)
		target = torch.zeros(batch_size,ques_len+2)
		for k = 1,batch_size do

			local ind = shuffle[k+(j-1)*batch_size]
			if j == 3363 then
				print(ind,12345)
			end	
			data1[k] = data_1['images_train'][shuffle[k+(j-1)*batch_size]]:resize(1,caption_input)
			data2[{{k},{2,ques_len+1}}] = data['train_questions'][shuffle[k+(j-1)*batch_size]]:resize(1,ques_len)
			data2[k][1] = vocab['word2ind']['<START>']
			target[k][1] = vocab['word2ind']['<START>']
			local lent = length(data['train_questions'][shuffle[k+(j-1)*batch_size]])
			target[{{k},{2,lent+1}}] = data['train_questions'][{{ind},{1,lent}}]:resize(1,lent)
			target[k][lent+2] = vocab['word2ind']['<END>']  		
		end
		--print(target,data2)
		if j == 3363 then
			print(j)
		end	
		data1  = data1:cuda()
		data2 = data2:cuda()
		target = target:cuda()
		if j == 3363 then
			print(data2,target)
			print(j)
		end
		for_data1  = model1:forward(data1)
		for_data2  = model2:forward(data2)
		if j == 3363  then
			print(data1)
			print(for_data1)

			print(j)
		end

		lstm_input = torch.CudaTensor(batch_size,ques_len+2,embedding_size)
		--lstm_input = lstm_input:cuda()
		if j == 3363 then
			print(j)
		end
		for t = 1,batch_size do
			lstm_input[t] = torch.cat(for_data1[t]:resize(1,embedding_size),for_data2[t]:resize(ques_len+1,embedding_size),1)
		end
		if j == 3363 then
			print(lstm_input)
			print(j)

		end
		--print(lstm_input:size())
		lstm_output = LSTM:forward(lstm_input)

		if j == 3363 then
			print(j)
		end

		-- if j == 2 and i == 1 then
		-- 	print(lstm_output)
		-- end
		if j == 3363 then
			print('heyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
			print(lstm_output,target)
		end	
	
		loss = criterion:forward(lstm_output,target)
		--print(lstm_ouput,target)

		if j == 3363 then
			print(j)
		end
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
		dm2 = dl[{{},{2,ques_len+2},{}}]
		
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
	if i%10 == 0 then
		torch.save('models/model_1_epoch'..tostring(i),model1)
		torch.save('models/model_2_epoch'..tostring(i),model2)
		torch.save('models/model_epoch'..tostring(i),model)
	end	
	print(loss_e,i)
	sample(index)
end	

-- --print(type(vocab['word2ind']['man']))

-- --sample(1)

	
	
