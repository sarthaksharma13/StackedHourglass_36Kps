require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'nngraph'
require 'valeval.lua'

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

model_file = '/home/sarthaksharma/anshuman/model_2.t7'
model = torch.load(model_file)
model:cuda()
model = model:get(1)
print('\nModel Loading Done')

iters = 0
preds = {}

img_path = '/tmp/synthia-sf/SYNTHIA-SF/SEQ4/RGBLeft/0000067.png'
coords = {915, 437, 1057, 561}

full_img = torch.FloatTensor(image.load(img_path))
img = image.crop(full_img, coords[1], coords[2], coords[3], coords[4])
img = image.scale(img, 64, 64)
input = torch.FloatTensor(1, 3, 64, 64)
input[1] = img:sub(1,3,1,64,1,64)

output = model:forward(input:cuda())
if type(output) == 'table' then
   output = output[#output]
end

keypoints = post_process(output, output_res)
kps = keypoints[1]:sub(1,36,3,4)
table.insert(preds, keypoints[1])

str = ''
for i=1,kps:size(1) do
   for j=1,kps:size(2) do
      str = str .. tostring(kps[i][j]) .. ' '
   end
end
str = string.sub(str, 1, #str-1)

print(str)
local f = io.open('output.txt', 'a')
f:close()
