Program design
- class CKY to implement the algorithm
- structure as follows:

class CKY()
    self.sentences: test file
    self.pcfg: grammar file

    def parse(): # method to parse all sentences
        for each sentence in sentences call self._parse_one()
        call self._output()

    def _parse_one(): # method to parse one sentences

    def _output(): format output


How to run？
- from the .py file's directory, run "python cky.py pcfg.txt" or "python cky.py pcfg.txt -prob"

Where did I text the program?
- on CADE lab1-22

Anything else?
- Try running "python cky.py pcfg.txt -verbose" or "python cky.py pcfg.txt -prob -verbose", there is a BONUS!
