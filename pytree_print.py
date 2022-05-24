def pytree_print(td, leaves):
    def node_visit(xs,node_data):
      if not node_data:
        labels = range(len(xs))
      elif isinstance(node_data, list):
        labels = node_data
      elif isinstance(node_data, xla_extension.PyTreeDef):
        out = node_data.walk(lambda x,d: d, lambda x:None, range(node_data.num_leaves))
        assert len(xs) == len(out) # if not, probably fine to choose out[-n:-1], but may be O(n^2) in tree size
        labels = out
      else:
        assert False

      return [*zip(map(str,labels),xs)]

    def print_with_paths(prefix, nodes):
      for (l,x) in nodes:
        p = prefix + '/' + l
        if isinstance(x, list):
          print_with_paths(p, x)
        else:
          print(p + ':', str(x[0]).replace('\n','\\n'))

    out = td.walk(node_visit, lambda x:(x,), leaves)
    print_with_paths('', out)

