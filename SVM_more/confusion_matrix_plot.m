cMatrix_percentage = zeros(size(cMatrix,1), size(cMatrix,2));

sum_1 = cMatrix(1,1) + cMatrix(1,2) + cMatrix(1,3);
sum_2 = cMatrix(2,1) + cMatrix(2,2) + cMatrix(2,3);
sum_3 = cMatrix(3,1) + cMatrix(3,2) + cMatrix(3,3);

cMatrix_percentage(1,:) = cMatrix(1,:) ./sum_1;
cMatrix_percentage(2,:) = cMatrix(2,:) ./sum_2;
cMatrix_percentage(3,:) = cMatrix(3,:) ./sum_3;

cMatrix_percentage = cMatrix_percentage.*100;

figure;
label = ['Cell  ';'Debris';'Strip '];
cellstr(label);
imagesc(cMatrix);            %# Create a colored plot of the matrix values
xlabel('Predicted label');
ylabel('True label');
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)
                         
textStrings = num2str(cMatrix(:));  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding

textStrings2 = num2str(cMatrix_percentage(:),'%0.2f%%');  %# Create strings from the matrix values
textStrings2 = strtrim(cellstr(textStrings2));  %# Remove any space padding


[x,y] = meshgrid(1:size(cMatrix));
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings of cMatrix
                'HorizontalAlignment','center');
hStrings2 = text(x(:),y(:)+0.15,textStrings2(:),...      %# Plot the strings of cMatrix_percentage
                'HorizontalAlignment','center');

midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = cMatrix(:) > midValue;  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color

textColor = cell(9,1);

for i = 1:size(textColors,1)
    if textColors(i) == 0
        textColor{i} = ['k'];
    else
        textColor{i} = ['w'];
    end
end

set(hStrings,{"color"},textColor);  %# Change the text colors
set(hStrings2,{"color"},textColor);  %# Change the text colors

set(gca,'XTick',1:size(cMatrix),...                         %# Change the axes tick marks
        'XTickLabel',label,...  %#   and tick labels
        'YTick',1:size(cMatrix),...
        'YTickLabel',label,...
        'TickLength',[0 0]);
