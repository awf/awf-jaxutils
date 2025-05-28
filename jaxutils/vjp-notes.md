So:
```
 foo(cos(x), g(x, y))
```
becomes
```
 let
   d1, d2 = vjp[foo](cos(x), g(x, y), dret)
 in mungle{ # D[cos(x), d1] o+ D[g(x, y), d2]
   # D[cos(x), d1]
   { dx : cos_vjp(x, d1) }
   oplus
   # D[g(x, y), d2]
   { dx, dy: g_vjp(x, y, d2) }
 }
```

How to represent hash table with tuple lhs?
1. Duplicate entries (but typically sharing the expr)
```
   { dx: g_subscript(g_vjp(x, y, d2), 0)
     dy: g_subscript(g_vjp(x, y, d2), 1)}
```
   Feels a bit dependent on getting the sharing
2. Explicit sharing through a dummy
```
   { _d123: g_vjp(x, y, d2),
     dx: _d123[0],
     dy: _d123[1]}
```
3. Just make it not a dict
```
    [dx, dy], g_vjp(x, y, d2)
```
   and we have a data structure for that, called Eqn
```
    Eqn([dx, dy], g_vjp(x, y, d2))
```

So:
```
 foo(cos(x), g(x, y))
```
becomes
```
 let
   d1, d2 = foo_vjp(cos(x), g(x, y), dret)
 in
   let
     dx_0 = cos_vjp(x, d1)
     dx_1, dy = g_vjp(x, y, d2)
   in
     g_tuple(dx_0 + dx_1, dy)
```

So:
```
 D[foo(cos(x), g(x, y)), dret] =
```
becomes
```
 [
   _d19_1, _d19_2 = foo_vjp(cos(x), g(x, y), dret)
   [
     _d1a = cos_vjp(x, _d19_1)
     [
       dx = _d1a
     ]
   ],
   [
     _d1b_1, _d1b_2 = g_vjp(x, y, _d19_2)
     [
       dx = _d1b_1
       dy = _d1b_2
     ]
   ]
 ]
```

So:
```
   D[foo(cos(x), g(x, y)), dret] =
```
becomes
```
 dx, dy =
    let
       _d19_1, _d19_2 = foo_vjp(cos(x), g(x, y), dret)
    dx_1 =
       in let
             _d1a = cos_vjp(x, _d19_1)
          in
             _d1a
    dx_2, dy_2 =
        let
          _d1b_1, _d1b_2 = g_vjp(x, y, _d19_2)
        in
          tuple(_d1b_1, _d1b_2)
    dx = dx_1 + dx_2
    dy = dy_2
 in
    g_tuple(dx, dy)
```
