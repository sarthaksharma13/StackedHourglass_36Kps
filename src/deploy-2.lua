require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'nngraph'
require 'valeval.lua'

data_file = '/home/sarthaksharma/anshuman/temp.txt'
model_file = '/home/sarthaksharma/anshuman/model_2.t7'

function get_predictions(heat_maps)
   assert(heat_maps:size():size() == 4, 'Input must be 4-D tensor')

   local elem, idx = torch.max(heat_maps:view(heat_maps:size(1), heat_maps:size(2), heat_maps:size(3)*heat_maps:size(4)), 3)
   local preds = torch.repeatTensor(idx, 1, 1, 2):float()

   preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % heat_maps:size(4) + 1 end)
   preds[{{}, {}, 2}]:add(-1):div(heat_maps:size(3)):floor():add(1)

   return preds
end

function post_process(output, output_res)
   local preds = get_predictions(output)
   local scores = torch.zeros(preds:size(1), preds:size(2), 1)

   for i=1,preds:size(1) do
      for j=1,preds:size(2) do
         local heat_map = output[i][j]
         local pred_x, pred_y = preds[i][j][1], preds[i][j][2]

         scores[i][j] = heat_map[pred_x][pred_y]
         if pred_x > 1 and pred_x < output_res and pred_y > 1 and pred_y < output_res then
            local diff = torch.Tensor({heat_map[pred_y][pred_x+1]-heat_map[pred_y][pred_x-1], heat_map[pred_y+1][pred_x]-heat_map[pred_y-1][pred_x]})
            preds[i][j]:add(diff:sign():mul(.25))
         end
      end
   end
   preds:add(0.5)

   return preds:cat(preds, 3):cat(scores, 3)
end

function accuracy(output,label)
   if type(output) == 'table' then
      return heatmapAccuracy(output[#output],label[#output],nil,dataset.accIdxs)
   else
      return heatmapAccuracy(output,label,nil,dataset.accIdxs)
   end
end

function split(str, dlm)
   i = 1;
   local res = {};
   for mat in string.gmatch(str, "([^"..dlm.."]+)") do
      res[i] = mat;
      i = i + 1;
   end
   return res;
end

torch.setdefaulttensortype('torch.FloatTensor')

num_stacks = 2
num_keypoints = 36
output_res = 64
pred_dims = {num_keypoints, 5}
input_dims = {3, 64, 64}

output_dims = {}
for i=1,num_stacks do
   output_dims[i] = {num_keypoints, 64, 64}
end

num_images = 0
for line in io.lines(data_file) do
   num_images = num_images + 1
end

model = torch.load(model_file)
model:cuda()
model = model:get(1)
print('\nModel Loading Done')

iters = 0
preds = {}
local f = io.open('keypoints-2.txt', 'a')
for line in io.lines(data_file) do
   iters = iters + 1

   str = split(line, " ")
   img_path = str[1]
   x = tonumber(str[2])
   y = tonumber(str[3])
   w = tonumber(str[4])
   h = tonumber(str[5])

   full_img = torch.FloatTensor(image.load(img_path))
   img = image.crop(full_img, x, y, x+w, y+h)
   img = image.scale(img, 64, 64)

   input = torch.FloatTensor(1, 3, 64, 64)
   input[1] = img

   output = model:forward(input:cuda())
   if type(output) == 'table' then
      output = output[#output]
   end

   keypoints = post_process(output, output_res)
   coords = keypoints[1]:sub(1,36,3,4)
   table.insert(preds, keypoints[1])

   str = ''
   for i=1,coords:size(1) do
      for j=1,coords:size(2) do
         str = str .. tostring(coords[i][j]) .. ' '
      end
   end
   str = string.sub(str, 1, #str-1)

   print('Done ' .. line)
   f:write(str)
end

f:close()
