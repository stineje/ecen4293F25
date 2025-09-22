# crypto with big ingegers

Starting with Python 3.11, the core developers added a guardrail against denial-of-service attacks (see PEP 670 and bpo-45235). <br>
* Converting a giant integer (millions of digits long) into a decimal string is relatively expensive: Python must repeatedly divide by 10 and collect remainders. <br>
* In a web server or REPL exposed to untrusted input, if someone feeds a 10-million-digit integer, the CPU can get stuck converting it to a string, which is a DoS vector. <br>
* To prevent this, Python 3.11 introduced a default cap on int → str and str → int conversions. <br>





