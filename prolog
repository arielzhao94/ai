prolog
fact: assert(parent(a,b)).
rule: assert((grandparent(X,Z) :- parent(X,Y),parent(Y,Z))).
delete a rule: retract().
connectives:
  conjunction: ,
  disjunction: ;
  negation: \+ Goal.

is/2
  used for arithmetic evaluation
  unlike regular unification(=)(just unifies values), is evaluates the right-hand side (evaluate expressions) before assigning it to the left-hand side
  only assigns values to variables, not constants
  `X is 5 + 3.`
  `square(X,Y) :- Y is X * X.
  square(4, Result).`

P31 exercise
sumOddEven(1,1).
sumOddEven(2,2).
sumOddEven(N,S) :- N >2, N1 is N - 2, sumOddEven(N1, Temp), S is N + Temp.

use call/1 to call a predicate
  `abstract :- nl, write('Enter X: '), read(X), positive(X).
  positive(X) :- X < 0, Y is -X, write(Y).
  positive(X) :- X >= 0, write(X).
  call(abstract).`
  when entering value as argument, end with `.`

P70 
p(X, 0, 1).
p(X, Y, Z) :- Y > 0, YY is Y - 1, p(X, YY, Temp), Z is X * Temp.

P80
f(1, 1).
f(2, 1).
f(N, R) :- N > 2, N1 is N - 1, N2 is N - 2, f(N-1, R1), f(N-2, R2), R is R1 + R2.
