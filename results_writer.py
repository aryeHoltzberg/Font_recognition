import csv

class Writer():
    def __init__(self,name:str = 'tmp',header:list = None):
        if not name.endswith('.csv'):
            name += '.csv'
        self.file = open(name,'w',newline='')
        self.writer = csv.writer(self.file)
        if header is not None:
            self.writer.writerow(header)
        self.lines_count = 0
    
    def write_line(self, line:list):
        self.writer.writerow(line)
        self.lines_count += 1
    
    def write_lines(self, lines:list):
        self.writer.writerow(lines)
    def close(self):
        self.file.close()
    
class ResultsWriter(Writer):
    header = [" ","image","char","Open Sans","Sansation","Titillium Web","Ubuntu Mono","Alex Brush"]
    def __init__(self, name: str = 'tmp', header: list = header):
        super().__init__(name,header)

    def __label_converter(self, index):
        label_mapping = {
            0:2,
            1:4,
            2:0,
            3:1,
            4:3,
        }
        return label_mapping[index.item()]
    def __create_row(self,img_name,char,index):
        indexes = [0,0,0,0,0]
        indexes[index] = 1
        line_number = self.lines_count
        row = [line_number,img_name,char]+indexes
        return row


    def _add_row(self,img_name,char,index):
        row = self.__create_row(img_name,char,index)
        self.write_line(row)     

    def add_word(self,img_name,word,label):
        index = self.__label_converter(label)
        for char in word:
            self._add_row(img_name,char,index)
            

if __name__ == '__main__':
    r = ResultsWriter('tmp2')
    r.add_word('img_1.jpg', 'bannan', 0)
    r.close()