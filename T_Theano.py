import theano
import theano.tensor as T
# x = T.dmatrix('x')
# s = 1/(1+T.exp(-x))
# logistic = theano.function([x],s)
# print logistic([[0,1],[-1,-2]])
# [[ 0.5         0.73105858]
# [ 0.26894142  0.11920292]]

# a,b = T.dmatrices('a','b')
# diff = a-b
# abs_diff = abs(a-b)
# diff_squre = diff**2
# f = theano.function([a,b],[diff,abs_diff,diff_squre])
# print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
# [array([[1., 0.],
#         [-1., -2.]]), array([[1., 0.],
#                              [1., 2.]]), array([[1., 0.],
#                                                 [1., 4.]])]

# x,y = T.scalars('x','y')
# z = x+y
# f = theano.function([x,theano.In(y,value=1)],z)
# print f(33) # 34.0
# print f(33,2) # 35.0

# from theano import shared
# state = shared(0)
# inc = T.iscalar('inc')
# accumulator = theano.function([inc],state,updates=[(state,state+inc)])
# decrementor = theano.function([inc], state, updates=[(state, state-inc)])

# print state.get_value() #0
# accumulator(1)
# print state.get_value() #1
# accumulator(200)
# print state.get_value() #201
#
# state.set_value(-9)
# accumulator(8)
# print state.get_value() #-1
# print state.type

# fn_of_state = state * 2 + inc
# foo = T.scalar(dtype=state.dtype)
# skip_shared = theano.function([inc, foo], fn_of_state, givens=[(state, foo)])
# print skip_shared(2, 3) # 8
# print skip_shared(4, 3) # 10
# print skip_shared(-1, 3) # -5
#
# state = theano.shared(0)
# inc = T.iscalar('inc')
# accumulator = theano.function([inc], state, updates=[(state, state+inc)])
# accumulator(10)
# print(state.get_value()) # 10
# new_state = theano.shared(0)
# new_accumulator = accumulator.copy(swap={state:new_state})
# new_accumulator(100)
# print(new_state.get_value()) # 100
# print(state.get_value()) # 10
