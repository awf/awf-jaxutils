
## fully bound lambdas 

This may be a read herring, may need to solve lambdas with freevars to see the good way

### Case1. Lambdas are like ints

Lambdas are not reals, they are more like strings or ints.
Their derivative contributions/update is the null set.
If they have freevars, we will explicitly make closures where the freevars are 
explicit (like g_scan takes freevars, we can automate that transformation for other fns) 

In the same way that
```
D[foo(7,x,x), dret] = 
    [x],  let
            _d1,_d2,_d3 = vjp(foo)(7,x,x,dret)
            _  += _d1  # dfreevars of arg1 (7)
            dx += _d2  # dfreevars of arg2 (x)
            dx += _d3  # dfreevars of arg3 (x)
          in [dx]
```
then given higher-order function `hof`, e.g.
```
def hof(f, x, y):
    p = f(x)
    q = f(y)
    ret = atan2(p, q)
    return ret
```
it's handled just like 7
```
D[hof((lambda t: e),x,x), dret] =
    [x],   let
               _d1,_d2,_d3 = vjp(hof)((lambda t: e),x,x,dret)
               _ += _d1  # dfreevars of arg1 (lambda t: e)
               dx += _d2  # dfreevars of arg2 (x)
               dx += _d3  # dfreevars of arg3 (x)
           in [dx]

def vjp(hof)(f, x, y, dret):
    # 1. Compute vjp of incoming function f
    # In general, this must be done at runtime, because f may be a newly consed 
    # up lambda.  And lambdas are just ints or strings, this is an inty operation
    f_vjp = vjp(f) 

    # 2. Forward pass, using f
    p = f(x)
    q = f(y)

    # 2. Bwd pass, using f_vjp
    dp,dq = atan2_vjp(p, q, dret)
    dx = f_vjp(x, dp)
    dy = f_vjp(x, dq)
    return (None, dx, dy)
```

### Case1.1.  Pass vjps with lambdas
But maybe we can say it's the caller's responsibility to supply the vjp with the f,
so that any vjp call may assume its fs are paired with their dfs.



```
 D[
   let
     v1 = e1
     f = lambda xs: e
     v2 = foo(v1, f, g)
   in
     body<v1,v2,g>
   dret
]
```
becomes
```
 [fvs(e)],
  let
    v1 = e1
    f, df = (lambda xs: e), (make_vjp(f))
    vs2 = foo(vs1, f) 

    dv1,dv2,dg += D[body, dret]  # map fvs of body (v1,v2,g) to contribs
    dv1,df,dg  += dfoo(vs1, (f, df), g, dv2) # lambdas in call are paired

    dfvs_e1 = D[e1, dvs1]   # mapping from freevars of e1 to diffs
  in
    osum(dfvs_body, dfvs_e2, dfvs_e1)
```

## fully bound lambdas 

This may be a red herring, may need to solve lambdas with freevars to see the good way

```
 D[
   let
     v1 = e1
     f = lambda xs: e
     v2 = foo(v1, f, g)
   in
     body<v1,v2,g>
   dret
]
```
becomes
```
 [fvs(e)],
  let
    v1 = e1
    f, df = (lambda xs: e), (make_vjp(f))
    vs2 = foo(vs1, f) 

    dv1,dv2,dg += D[body, dret]  # map fvs of body (v1,v2,g) to contribs
    dv1,df,dg  += dfoo(vs1, (f, df), g, dv2) # lambdas in call are paired

    dfvs_e1 = D[e1, dvs1]   # mapping from freevars of e1 to diffs
  in
    osum(dfvs_body, dfvs_e2, dfvs_e1)
```
