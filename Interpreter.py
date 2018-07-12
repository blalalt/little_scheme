
def tokenize(text):
    """
    :param text: 源程序字符串
    :return: 词法单元组成的列表
    """
    current_token = ''
    tokens = []

    for ch in text:
        if ch.isspace():
            # 当遇到 空格或者括号时表明上一个词法单元的结束
            if (len(current_token) > 0):
                tokens.append(current_token)
                current_token = ''
        elif ch in '()':
            if len(current_token) > 0:
                tokens.append(current_token)
                current_token = ''
            tokens.append(ch)
        else:
            current_token += ch
    if len(current_token) > 0:
        tokens.append(current_token)
    return tokens


def construct_tokens(tokens: list):
    res = []
    while tokens:
        current_token:str = tokens.pop(0)
        if current_token == '(':
            res.append(construct_tokens(tokens))
        elif current_token == ')':
            return res
        else: # token 是数值或者符号
            obj = None
            if current_token.isdigit():
                obj = int(current_token)
            else:
                try:
                    obj = float(current_token)
                except ValueError:
                    obj = current_token
            res.append(obj)
    return res

def parser(text):
    return construct_tokens(tokenize(text))

class Environment(dict):
    def __init__(self, parent):
        super(Environment, self).__init__()
        self.parent = parent

    def bind(self, symbol, value):
        """
        作用域内定义变量或函数
        :param symbol:变量或函数名
        :param value:值
        :return:None
        """
        self[symbol] = value

    def find(self, symbol):
        """
        查找符号的值，查找规则是:local -> 上一个作用域->...->Global->Builtin
        :param symbol:
        :return:
        """
        env = self
        while env != None:
            if symbol in env.keys():
                return env[symbol]
            else:
                env = env.parent
        raise LookupError('Not found Symbol: {}'.format(symbol))

    def make_child_env(self, params:list, args:list):
        child_env = Environment(self)
        for k, v in zip(params, args):
            child_env.bind(k ,v)
        return child_env

def get_builtin_env():
    env = Environment(None)

    import operator
    import math
    env.update(
        {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.add,
            '/': operator.truediv,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            'nil': None,
            '#t': True,
            '#f': False,
            'true': True,
            'false': False,
            'number?': is_number,
            'string?': is_string,
            'display': print,
        }
    )
    return env


def is_proceduce(symbol):
    return callable(symbol)

def is_symbol(symbol):
    return isinstance(symbol, str) and not symbol.startswith('"')

def is_number(symbol):
    return isinstance(symbol, (int, float))

def is_string(symbol):
    return isinstance(symbol, str) and symbol.startswith('"')

built_in_env = get_builtin_env()
global_env = Environment(built_in_env)


def define_proceduce(args, env=global_env):
    sym, expr = args
    if isinstance(sym, list): #(define (func_name args..) (body))
        sym, params = sym[0], sym[1:]
        obj = Proceduce(params, expr, env)
    else: # (define symbol expr)
        obj = evaluate(expr, env)
    env.bind(sym, obj)
    return sym

def if_proceduce(args, env=global_env):
    cond, true_s, false_s = args
    expr = (true_s if evaluate(cond, env) else false_s)
    return evaluate(expr, env)

def set_proceduce(args, env=global_env):
    symbol, expr = args
    env.bind(symbol, evaluate(expr, env))
    return symbol

def quote_proceduce(args, env=global_env):
    return args[0]

def lambda_proceduce(args, env=global_env):
    params, body = args
    return Proceduce(params, body, env)

def begin_proceduce(args, env=global_env):
    return [evaluate(expr, env) for expr in args][-1]

def cond_proceduce(args, env=global_env):
    for statement in args:
        cond, expr = statement
        if cond == 'else':
            return evaluate(expr, env)
        else:
            res = evaluate(cond, env)
            if res: return res
            else: continue


def and_proceduce(args, env=global_env):
    if len(args) == 0:
        return True
    res = None
    for cond_expr in args:
        res = evaluate(cond_expr, env)
        if not res: return False
    else:
        return res

def or_proceduce(args, env=global_env):
    if len(args) == 0:
        return False
    res = None
    for cond_expr in args:
        res = evaluate(cond_expr, env)
        if res: return res
    else:
        return False

def let_proceduce(args, env=global_env):
    bind_exprs, body = args
    params, args = [], []
    for bind_expr in bind_exprs:
        symbol, value = bind_expr
        params.append(symbol)
        args.append(evaluate(value, env))
    new_env = env.make_child_env(params, args)
    return evaluate(body, new_env)

SPECIAL_FORMS = {'define': define_proceduce, 'if': if_proceduce, 'quote': quote_proceduce,
                 'set!': set_proceduce, 'lambda': lambda_proceduce, 'begin': begin_proceduce,
                 'and': and_proceduce, 'or': or_proceduce, 'cond': cond_proceduce, 'let': let_proceduce}


def evaluate(expr, env=global_env):
    """
    :param expr: 要执行的表达式
    :param env: 作用域
    :return: 表达式执行结果
    """
    # 原子表达式 如： a, "123", 1.2等
    if is_symbol(expr):
        return env.find(expr)
    if is_number(expr) or is_string(expr) or expr is None:
        return expr

    # 组合表达式
    if not isinstance(expr, list):
        raise SyntaxError("SyntaxError")

    op, remainder = expr[0], expr[1:]

    if is_symbol(op) and op in SPECIAL_FORMS: # 特殊形式的方法
        return SPECIAL_FORMS[op](remainder, env)
    else:
        proc = evaluate(op, env)
        operands = [evaluate(exp, env) for exp in remainder]
        if not is_proceduce(proc):
             raise TypeError("{} is not callable.".format(proc))
        return proc(*operands)

class Proceduce:
    """
    用户自定义的Scheme函数的解释器内部python表示
    """
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env

    def __call__(self, *args, **kwargs):
        return evaluate(self.body,
                        self.env.make_child_env(self.params, args))

def read_eval_print_loop():
    prompt = 'LiteScheme>> '
    while True:
        exprs = parser(input(prompt))
        res = [evaluate(expr) for expr in exprs]
        print(res[-1])

if __name__ == '__main__':
    read_eval_print_loop()

