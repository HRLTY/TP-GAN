function [img_cropped,trans_pt]  = face_db_align_single_custom(img, keypoints)

% center of eyes (ec), center of l&r mouth(mc), rotate and resize
% ec_mc_y: y_mc-y_ec, diff of height of ec & mc, to scale the image.
% ec_y: top of ec, to crop the face.
% Dataset	size	ec_mc_y	ec_y
% Training set	144x144	48	48
% Testing set	128x128	48	40
ec_mc_y = 48; ec_y = 40; img_size = 128;
%ec_mc_y = 96; ec_y = 80; img_size = 256;
crop_size = img_size;

 [img_cropped, trans_pt]...
                = align(img, keypoints, crop_size, ec_mc_y, ec_y);   
%         figure(1);
%         subplot(1,3,1);
%         imshow(img);
%         hold on;
%         plot(f5pt(:,1),f5pt(:,2), 'bo');
% %         plot(f5pt(2,1),f5pt(2,2), 'bo');
% %         plot(f5pt(3,1),f5pt(3,2), 'bo');
% %         plot(f5pt(4,1),f5pt(4,2), 'bo');
% %          plot(f5pt(5,1),f5pt(5,2), 'bo');
%         hold off;
%         subplot(1,3,2);
%         imshow(img2);
%         rectangle('Position', [round(eyec(1)) round(eyec(2)) 10 10]);
%         hold on;
%         plot(eyec(1), eyec(2), 'ro');
%         plot(10,100, 'bx');
%         hold off;
%         subplot(1,3,3);
%         imshow(img_cropped);
%         hold on;
%         plot(trans_points(:,1),trans_points(:,2), 'bo');
%         hold off;
%         figure(2);
%         subplot(2,2,1);
%         imshow(eyel_crop);
%         subplot(2,2,2);
%         imshow(eyer_crop);
%         subplot(2,2,3);
%         imshow(nose_crop);
%         subplot(2,2,4);
%         imshow(month_crop);
%         pause;
%      
        
%        img_final = imresize(img_cropped, [img_size img_size], 'Method', 'bicubic');
%         if size(img_final,3)>1
%             img_final = rgb2gray(img_final);
%         end
%         save_fn = [save_dir '/' filename(1:end-4) '_cropped.png'];
%         imwrite(img_cropped, save_fn);

end

function res = read_5pt(fn)
fid = fopen(fn, 'r');
raw = textscan(fid, '%f %f');
fclose(fid);
res = [raw{1} raw{2}];
end

function [cropped, trans_points] ...
    = align(img, f5pt, crop_size, ec_mc_y, ec_y)
f5pt = double(f5pt);
if f5pt(1,1) == f5pt(2,1)
    ang = 0;
else
    ang_tan = (f5pt(1,2)-f5pt(2,2))/(f5pt(1,1)-f5pt(2,1));
    ang = atan(ang_tan) / pi * 180;
end
% if abs(ang) > 10 || abs(f5pt(1,1) - f5pt(2,1)) < 10
%     %fprintf('damn, ang:%.0f, distance:%d\n',round(ang), abs(f5pt(1,1) - f5pt(2,1)))
%     ang = ang / 5;
% %else
%    %fprintf('good, ang:%.0f, distance:%d\n',ang, abs(f5pt(1,1) - f5pt(2,1)))
% end
img_rot = imrotate(img, ang, 'bicubic');
imgh = size(img,1);
imgw = size(img,2);

% eye center
x = (f5pt(1,1)+f5pt(2,1))/2;
y = (f5pt(1,2)+f5pt(2,2))/2;
% x = ffp(1);
% y = ffp(2);

ang = -ang/180*pi;
%{
x0 = x - imgw/2;
y0 = y - imgh/2;
xx = x0*cos(ang) - y0*sin(ang) + size(img_rot,2)/2;
yy = x0*sin(ang) + y0*cos(ang) + size(img_rot,1)/2;
%}
[xx, yy] = transform(x, y, ang, size(img), size(img_rot));
eyec = round([xx yy]);
% eyel = round(transform(f5pt(1,1),f5pt(1,2), ang, size(img), size(img_rot)));
% eyer = round(transform(f5pt(2,1),f5pt(2,2), ang, size(img), size(img_rot)));
% nose = round(transform(f5pt(3,1),f5pt(3,2), ang, size(img), size(img_rot)));
tem = size(f5pt);
if tem(1) == 5
    x = (f5pt(4,1)+f5pt(5,1))/2;
    y = (f5pt(4,2)+f5pt(5,2))/2;
elseif tem(1) == 4
    x = f5pt(4,1);
    y = f5pt(4,2);
end
[xx, yy] = transform(x, y, ang, size(img), size(img_rot));
mouthc = round([xx yy]);

resize_scale = ec_mc_y/(mouthc(2)-eyec(2));
if resize_scale < 0
    resize_scale = 1
end
%resize_scale = resize_scale
img_resize = imresize(img_rot, resize_scale);

%res = img_resize;
eyec2 = (eyec - [size(img_rot,2)/2 size(img_rot,1)/2]) * resize_scale + [size(img_resize,2)/2 size(img_resize,1)/2];
eyec2 = round(eyec2);

%% build trans_points centers for five parts

trans_points = zeros(size(f5pt));
[trans_points(:,1),trans_points(:,2)] = transform(f5pt(:,1),f5pt(:,2), ang, size(img), size(img_rot));
trans_points = round(trans_points);
trans_points(:,1) = trans_points(:,1) - size(img_rot,2)/2;
trans_points(:,2) = trans_points(:,2) - size(img_rot,1)/2;
trans_points =  trans_points * resize_scale;
trans_points(:,1) = trans_points(:,1) + size(img_resize,2)/2;
trans_points(:,2) = trans_points(:,2) + size(img_resize,1)/2;
trans_points = round(trans_points);

img_crop = zeros(crop_size, crop_size, size(img_rot,3));
% crop_y = eyec2(2) -floor(crop_size*1/3);
crop_y = eyec2(2) - ec_y;
crop_y_end = crop_y + crop_size - 1;
crop_x = eyec2(1)-floor(crop_size/2);
crop_x_end = crop_x + crop_size - 1;

box = guard([crop_x crop_x_end crop_y crop_y_end], size(img_resize));
img_crop(box(3)-crop_y+1:box(4)-crop_y+1, box(1)-crop_x+1:box(2)-crop_x+1,:) = img_resize(box(3):box(4),box(1):box(2),:);
%calculate relative coordinate
trans_points(:,1) = trans_points(:,1) - box(1) + 1;
trans_points(:,2) = trans_points(:,2) - box(3) +1;
% img_crop = img_rot(crop_y:crop_y+img_size-1,crop_x:crop_x+img_size-1);
cropped = img_crop/255;
end

function r = guard(x, N)
x(x<1)=1;
if N(2)>0
    x(logical((x>N(2)) .* [1 1 0 0]))=N(2);
end
if N(1)
    x(logical((x>N(1)) .* [0 0 1 1]))=N(1);
end
x(x<1)=1;
r = x;
end

function [xx, yy] = transform(x, y, ang, s0, s1)
% x,y position
% ang angle
% s0 size of original image
% s1 size of target image

x0 = x - s0(2)/2;
y0 = y - s0(1)/2;
xx = x0*cos(ang) - y0*sin(ang) + s1(2)/2;
yy = x0*sin(ang) + y0*cos(ang) + s1(1)/2;
end