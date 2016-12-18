cMatrix_sum = sum(sum(cMatrix));
cMatrix_percentage = cMatrix./cMatrix_sum;
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
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
hStrings2 = text(x(:),y(:)+0.15,textStrings2(:),...      %# Plot the strings
                'HorizontalAlignment','center');

midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(cMatrix(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color

set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
set(hStrings2,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:size(cMatrix),...                         %# Change the axes tick marks
        'XTickLabel',label,...  %#   and tick labels
        'YTick',1:size(cMatrix),...
        'YTickLabel',label,...
        'TickLength',[0 0]);
