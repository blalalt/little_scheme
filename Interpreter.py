
def tokenize(text):
    current_token = '' # 保存当前词素
    tokens = [] # 保存每个词法单元
    for ch in text: # 对每个字符进行处理
        if ch.isspace(): # 空白
            if (len(current_token) > 0): # 空白也是一个分隔符,分隔符表示一个词法单元的结束
                tokens.append(current_token)
                current_token = ''
        elif ch in '()':
            if len(current_token) > 0: # 括号同样是一个分隔符
                tokens.append(current_token)
                current_token = ''
            tokens.append(ch)
        else:
            current_token += ch
    if len(current_token) > 0:
        tokens.append(current_token)
    return tokens

def construct_tokens(tokens):
    res = []
    while tokens:
        current_token = tokens.pop(0)
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
            'display': print,
        }
    )
    return env
built_in_env = get_builtin_env()
global_env = Environment(built_in_env)

class Proceduce:

    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env

    def __call__(self, *args, **kwargs):
        """
        1. 根据形参和实参以及过程定义的环境创建一个新的环境
        2. 在新的环境内执行函数体
        """
        return evaluate(self.body,
                        self.env.make_child_env(self.params, args))

def is_proceduce(symbol):
    return callable(symbol)

def is_symbol(symbol):
    return isinstance(symbol, str) and not symbol.startswith('"')

def is_number(symbol):
    return isinstance(symbol, (int, float))

def is_string(symbol):
    return isinstance(symbol, str) and symbol.startswith('"')



def if_proceduce(args, env=global_env):
    cond, true_s = args[:2]
    false_s = args[2] if len(args) > 2 else 'nil'
    expr = (true_s if evaluate(cond, env) is not False else false_s)
    return evaluate(expr, env)

def cond_proceduce(args, env=global_env):
    if args[:-1][0] == 'else':
        args[:-1][0] = '#t'
    for statement in args:
        cond, exprs = statement[0], statement[1:]
        res = evaluate(cond, env)
        if len(exprs) == 0: return res
        return [eval(expr, env) for expr in exprs][:-1]


def and_proceduce(args, env=global_env):
    if len(args) == 0:
        args = '#t'
    res = None
    for cond_expr in args:
        res = evaluate(cond_expr, env)
        if res is False: return False
    else:
        return res

def or_proceduce(args, env=global_env):
    if len(args) == 0:
        args = '#f'
    res = None
    for cond_expr in args:
        res = evaluate(cond_expr, env)
        if res is not False: return res
    else:
        return False

def define_proceduce(args, env=global_env):
    sym, expr = args
    if isinstance(sym, list): #(define (func_name args..) (body))
        sym, params = sym[0], sym[1:]
        obj = Proceduce(params, expr, env)
    else: # (define symbol expr)
        obj = evaluate(expr, env)
    env.bind(sym, obj)
    return sym

def begin_proceduce(args, env=global_env):
    return [evaluate(expr, env) for expr in args][-1]

def let_proceduce(args, env=global_env):
    bind_exprs, body = args
    params, args = [], []
    for bind_expr in bind_exprs:
        symbol, value = bind_expr
        params.append(symbol)
        args.append(evaluate(value, env))
    new_env = env.make_child_env(params, args)
    return evaluate(body, new_env)

def set_proceduce(args, env=global_env):
    symbol, expr = args
    if symbol in env.keys():
        env.bind(symbol, evaluate(expr, env))
    else: raise EnvironmentError("Not found Symbol: {}.".format(symbol))
    return symbol

def lambda_proceduce(args, env=global_env):
    params, body = args
    return Proceduce(params, body, env)

def quote_proceduce(args, env=global_env):
    return args[0]

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

    op, remainder = expr[0], expr[1:]  # 操作符合操作数分离

    # 用操作符来判断是否是 Special Forms
    if is_symbol(op) and op in SPECIAL_FORMS:  # 特殊形式的方法
        return SPECIAL_FORMS[op](remainder, env)
    else:
        proc = evaluate(op, env)  # 找到与操作符相关联的 `Proceduce`实例或者Python函数
        operands = [evaluate(exp, env) for exp in remainder]  # 求操作数
        if not is_proceduce(proc):
            raise TypeError("{} is not callable.".format(proc))
        return proc(*operands)

def read_eval_print_loop():
    prompt = 'LiteScheme>> '
    while True:
        text = input(prompt)
        if not text: continue
        exprs = parser(text)
        res = [evaluate(expr) for expr in exprs][-1] # 只返回最后一个表达式的执行结果
        if res: print(res)

if __name__ == '__main__':
    read_eval_print_loop()
