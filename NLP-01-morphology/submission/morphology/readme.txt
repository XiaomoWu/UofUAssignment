Which machine did I test on?
- lab1-22.eng.utah.edu

Any known problem?
- No

Brief of implementation: I created a class "PARSE", whose structure as follows:

class PARSE:
    def __ini__():
        # self.dict: dict file
        # self.rules: rules file
        # self.test: test file
    
    def parse():
        # loop over all words in self.test
        # for each word, call self._parse_one(word)
        # self._parse_one() is called RECURSIVELY

    def _parse_one(word):
        # method to parse one word

    def _OTHER_HELPER_METHODS():
        ......

