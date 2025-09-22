# crypto with big ingegers

## Starting with Python 3.11, the core developers added a guardrail against denial-of-service attacks (see PEP 670 and bpo-45235). <br>
* Converting a giant integer (millions of digits long) into a decimal string is relatively expensive: Python must repeatedly divide by 10 and collect remainders. <br>
* In a web server or REPL exposed to untrusted input, if someone feeds a 10-million-digit integer, the CPU can get stuck converting it to a string, which is a DoS vector. <br>
* To prevent this, Python 3.11 introduced a default cap on int → str and str → int conversions. <br>

See [PEP 670: Convert macros to functions in the Python C API](https://peps.python.org/pep-0670/).

## New Limit in Python 3.11+

Starting in **Python 3.11**, Python enforces a safeguard when converting very large integers
to or from strings. By default, there is a cap of **4,300 decimal digits**.

If you try to convert a larger integer, you’ll see:
```
ValueError: Exceeds the limit (4300) for integer string conversion;
use sys.set_int_max_str_digits() to increase the limit
```

## This affects operations like:

* `str(big_int)`
* `int("123...")` with a very long string

## How to Adjust or Disable the Limit

You can raise or remove the limit at the start of your script:

```python
import sys

# Allow up to 1 million digits in str() conversion
sys.set_int_max_str_digits(1_000_000)

# Or disable the limit entirely
sys.set_int_max_str_digits(0)


