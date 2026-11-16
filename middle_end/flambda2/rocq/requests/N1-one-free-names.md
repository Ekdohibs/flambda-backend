# N1: the statements about names, with the one function for free names

The series was prepared after the first step of the heap and the
passes for loops, and held while the epochs that change the machine
were moving. It was replayed over the validation that renames binders
apart, with no proof repaired, and is in. This is the request as it
was made, with the counts of the replay and with what has changed
since marked as such.

The decision to replace was the human's to take. He was away, and the
orchestrator took it overnight, for his review, as recommended in
`status/N1.md`: replace, with every older form kept as a lemma for as
long as a proof uses it, and with no statement made weaker. Section 8
says what it costs to go back.

Contents: 1 the two accounts; 2 what the series does; 3 each
statement, before and after; 4 the lemmas that keep the older forms;
5 what changes in files of other owners; 6 which form gives which;
7 what is still written in the older style, and why; 8 going back,
and going on; 9 validation; 10 findings.

## 1. The two accounts

The free names of a term are given by one function for each sort,
`free_variables_of_expr` and `free_continuations_of_expr`
(`FlambdaBindersSpec.v`). They go into code. The three conditions on
binders of well-formedness are written with them.

The older account is the relations `var_free_in_expr` and
`cont_free_in_expr` (`FlambdaNominalSpec.v`). They do not go into
code: a name that occurs only in the code of a term is not free by
them. The statements about binders and the rules of alpha-equivalence
were written with them, before the human ruled that code is closed
(D31) and the one function was written.

The two are related by `FlambdaFreeNamesSpec.v`, which stays: what is
free by the relation is among the list, whatever the code
(`var_free_among_statement`, `cont_free_among_statement`); and where
the code is closed the two say the same (`var_free_agrees_statement`,
`cont_free_agrees_statement`).

## 2. What the series does

The rule it follows: a statement is replaced only where the form with
the one function says the same. That is where the statement asks that
the code is closed.

- **Replaced.** The six statements of `FlambdaNominalSpec.v` that a
  bound name may be renamed are no longer statements of a `Spec` file.
  Their forms with the one function, which were proved beside them
  and reviewed (`FlambdaCanonicalFreeNamesSpec.v`), are the statements
  now. Their text is not changed by this series. Two statements of
  `FlambdaAlphaSpec.v` are replaced in the same way.
- **The rules.** The five rules of `alpha_equivalent` that rename
  (`FlambdaAlphaSpec.v`) take their premise about names with the one
  function. The relation defined is the one that was.
- **Not replaced, on purpose.** The three statements of
  `FlambdaNominalSpec.v` about a binding that is not used stay as they
  were, written with the relations. They ask nothing of the code, and
  with the one function they would say less (section 6); the rewrites
  of the validator that remove bindings are proved from them. Their
  forms with the one function stay beside them, as corollaries. The
  three statements of `FlambdaAlphaSpec.v` about the relations
  themselves stay too: where code is not closed they are other facts
  than those about the functions.
- **Deleted.** One statement of the second prover's
  (`FlambdaFreshBindersSpec.v`): it follows, on every term, from a
  statement about lists that was already there. And the five
  statements that said that the rules held in the newer form: they
  are the rules now.
- Each of the nine statements in an older form that leaves a `Spec`
  file (the six, the two of `FlambdaAlphaSpec.v` and the second
  prover's one) is kept, with its name and its text, as a lemma of the
  `Proofs` file that proved it. No proof that used one is changed.
  The five statements that said that the rules held in the newer form
  are not kept under their names: the constructors of
  `alpha_equivalent` are those claims now, nothing else in the tree
  used the five, and their theorems go with them.
- The relations stay defined, in `FlambdaNominalSpec.v`, under a
  heading that says what they are for (section 7).

The claims that `make claims` checks go from 2126 to 2113: fourteen
statements leave the `Spec` files (the six, the two, the five forms
of the rules, and the second prover's one) and one enters.

## 3. Each statement, before and after

The owner of every statement here is N1, except that of 3.4, which is
N2's. In each pair the conclusion is the same; what changes is how the
premise about names is said. `fixes_names r vs ks` says that `r` moves
none of the variables `vs` and none of the continuations `ks`; it is
defined in `FlambdaAlphaSpec.v` (it was in
`FlambdaCanonicalFreeNamesSpec.v`).

### 3.1 The names that a binder binds may be renamed

These were in `FlambdaNominalSpec.v`. The statements are now those of
`FlambdaCanonicalFreeNamesSpec.v`. Each asks that the code is closed.

**The variable of a `Let`.**
Before, `rename_let_statement`; it is now the lemma
`FlambdaNominalProofs.rename_let`:

```coq
Definition rename_let_statement : Prop :=
  ∀ (r : renaming) (x : variable) (kind : flambda_kind)
    (defining_expr : named) (body : expr),
    code_closed body = true ->
    (∀ y : variable,
      y <> x -> var_free_in_expr y body -> rename_variable r y = y) ->
    (∀ k : continuation,
      cont_free_in_expr k body -> rename_continuation r k = k) ->
    expr_equivalent (Let x kind defining_expr body)
      (Let (rename_variable r x) kind defining_expr (rename_expr r body)).
```

After, `rename_let_canonical_statement`:

```coq
Definition rename_let_canonical_statement : Prop :=
  ∀ (r : renaming) (x : variable) (kind : flambda_kind)
    (defining_expr : named) (body : expr),
    code_closed body = true ->
    fixes_names r (without [x] (free_variables_of_expr body))
      (free_continuations_of_expr body) ->
    expr_equivalent (Let x kind defining_expr body)
      (Let (rename_variable r x) kind defining_expr (rename_expr r body)).
```

**Opening a `Let` at a fresh variable.**
Before, `freshen_let_statement`; it is now the lemma
`FlambdaNominalProofs.freshen_let`:

```coq
Definition freshen_let_statement : Prop :=
  ∀ (x fresh : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    code_closed body = true ->
    ~ var_free_in_expr fresh body ->
    expr_equivalent (Let x kind defining_expr body)
      (Let fresh kind defining_expr
         (rename_expr (Renaming [(x, fresh)] []) body)).
```

After, `freshen_let_canonical_statement`:

```coq
Definition freshen_let_canonical_statement : Prop :=
  ∀ (x fresh : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    code_closed body = true ->
    among fresh (free_variables_of_expr body) = false ->
    expr_equivalent (Let x kind defining_expr body)
      (Let fresh kind defining_expr
         (rename_expr (Renaming [(x, fresh)] []) body)).
```

**The continuation of a non-recursive `Let_cont`.**
Before, `rename_let_cont_statement`; it is now the lemma
`FlambdaNominalProofs.rename_let_cont`:

```coq
Definition rename_let_cont_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (handler : continuation_handler)
    (body : expr),
    code_closed body = true ->
    (∀ y : variable, var_free_in_expr y body -> rename_variable r y = y) ->
    (∀ k' : continuation,
      k' <> k -> cont_free_in_expr k' body -> rename_continuation r k' = k')
    ->
    expr_equivalent (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive (rename_continuation r k) handler
         (rename_expr r body)).
```

After, `rename_let_cont_canonical_statement`:

```coq
Definition rename_let_cont_canonical_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (handler : continuation_handler)
    (body : expr),
    code_closed body = true ->
    fixes_names r (free_variables_of_expr body)
      (without [k] (free_continuations_of_expr body)) ->
    expr_equivalent (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive (rename_continuation r k) handler
         (rename_expr r body)).
```

**The parameters of a handler.**
Before, `rename_handler_params_statement`; it is now the lemma
`FlambdaNominalProofs.rename_handler_params`:

```coq
Definition rename_handler_params_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr),
    code_closed handler = true ->
    (∀ y : variable,
      y ∉ params -> var_free_in_expr y handler -> rename_variable r y = y) ->
    (∀ k' : continuation,
      cont_free_in_expr k' handler -> rename_continuation r k' = k') ->
    expr_equivalent
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k
         (rename_handler r (Handler params metadata handler)) body).
```

After, `rename_handler_params_canonical_statement`:

```coq
Definition rename_handler_params_canonical_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr),
    code_closed handler = true ->
    fixes_names r
      (free_variables_of_handler (Handler params metadata handler))
      (free_continuations_of_expr handler) ->
    expr_equivalent
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k
         (rename_handler r (Handler params metadata handler)) body).
```

**The parameters of a handler of a recursive `Let_cont`.**
Before, `rename_recursive_handler_params_statement`; it is now the lemma
`FlambdaNominalProofs.rename_recursive_handler_params`:

```coq
Definition rename_recursive_handler_params_statement : Prop :=
  ∀ (r : renaming) (before : list (continuation * continuation_handler))
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr)
    (after : list (continuation * continuation_handler)) (body : expr),
    code_closed handler = true ->
    (∀ y : variable,
      y ∉ params -> var_free_in_expr y handler -> rename_variable r y = y) ->
    (∀ k' : continuation,
      cont_free_in_expr k' handler -> rename_continuation r k' = k') ->
    expr_equivalent
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler) :: after) body)
      (Let_cont_recursive
         (before
          ++ (k, rename_handler r (Handler params metadata handler)) :: after)
         body).
```

After, `rename_recursive_handler_params_canonical_statement`:

```coq
Definition rename_recursive_handler_params_canonical_statement : Prop :=
  ∀ (r : renaming) (before : list (continuation * continuation_handler))
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr)
    (after : list (continuation * continuation_handler)) (body : expr),
    code_closed handler = true ->
    fixes_names r
      (free_variables_of_handler (Handler params metadata handler))
      (free_continuations_of_expr handler) ->
    expr_equivalent
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler) :: after) body)
      (Let_cont_recursive
         (before
          ++ (k, rename_handler r (Handler params metadata handler)) :: after)
         body).
```

**The continuations of a recursive `Let_cont`.**
Before, `rename_let_cont_recursive_statement`; it is now the lemma
`FlambdaNominalProofs.rename_let_cont_recursive`:

```coq
Definition rename_let_cont_recursive_statement : Prop :=
  ∀ (r : renaming) (handlers : list (continuation * continuation_handler))
    (body : expr),
    code_closed (Let_cont_recursive handlers body) = true ->
    (∀ y : variable,
      var_free_in_expr y (Let_cont_recursive handlers body) ->
      rename_variable r y = y) ->
    (∀ k : continuation,
      cont_free_in_expr k (Let_cont_recursive handlers body) ->
      rename_continuation r k = k) ->
    expr_equivalent (Let_cont_recursive handlers body)
      (rename_expr r (Let_cont_recursive handlers body)).
```

After, `rename_let_cont_recursive_canonical_statement`:

```coq
Definition rename_let_cont_recursive_canonical_statement : Prop :=
  ∀ (r : renaming) (handlers : list (continuation * continuation_handler))
    (body : expr),
    code_closed (Let_cont_recursive handlers body) = true ->
    fixes_names r
      (free_variables_of_expr (Let_cont_recursive handlers body))
      (free_continuations_of_expr (Let_cont_recursive handlers body)) ->
    expr_equivalent (Let_cont_recursive handlers body)
      (rename_expr r (Let_cont_recursive handlers body)).
```

### 3.2 The rules of `alpha_equivalent` (`FlambdaAlphaSpec.v`)

The constructors keep their names. The older form of each is the
lemma of `FlambdaAlphaProofs.v` with `_by_relation` after the name.

**`Alpha_rename`.** Before:

```coq
  | Alpha_rename (r : renaming) (e : expr) :
    code_closed e = true ->
    (∀ x : variable, var_free_in_expr x e -> rename_variable r x = x) ->
    (∀ k : continuation,
      cont_free_in_expr k e -> rename_continuation r k = k) ->
    alpha_equivalent e (rename_expr r e)
```

After:

```coq
  | Alpha_rename (r : renaming) (e : expr) :
    code_closed e = true ->
    fixes_names r (free_variables_of_expr e)
      (free_continuations_of_expr e) ->
    alpha_equivalent e (rename_expr r e)
```

**`Alpha_let`.** Before:

```coq
  | Alpha_let (r : renaming) (x : variable) (kind : flambda_kind)
      (defining_expr : named) (body : expr) :
    code_closed body = true ->
    (∀ y : variable,
      y <> x -> var_free_in_expr y body -> rename_variable r y = y) ->
    (∀ k : continuation,
      cont_free_in_expr k body -> rename_continuation r k = k) ->
    alpha_equivalent (Let x kind defining_expr body)
      (Let (rename_variable r x) kind defining_expr (rename_expr r body))
```

After:

```coq
  | Alpha_let (r : renaming) (x : variable) (kind : flambda_kind)
      (defining_expr : named) (body : expr) :
    code_closed body = true ->
    fixes_names r (without [x] (free_variables_of_expr body))
      (free_continuations_of_expr body) ->
    alpha_equivalent (Let x kind defining_expr body)
      (Let (rename_variable r x) kind defining_expr (rename_expr r body))
```

**`Alpha_let_cont`.** Before:

```coq
  | Alpha_let_cont (r : renaming) (k : continuation)
      (handler : continuation_handler) (body : expr) :
    code_closed body = true ->
    (∀ y : variable, var_free_in_expr y body -> rename_variable r y = y) ->
    (∀ k' : continuation,
      k' <> k -> cont_free_in_expr k' body -> rename_continuation r k' = k')
    ->
    alpha_equivalent (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive (rename_continuation r k) handler
         (rename_expr r body))
```

After:

```coq
  | Alpha_let_cont (r : renaming) (k : continuation)
      (handler : continuation_handler) (body : expr) :
    code_closed body = true ->
    fixes_names r (free_variables_of_expr body)
      (without [k] (free_continuations_of_expr body)) ->
    alpha_equivalent (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive (rename_continuation r k) handler
         (rename_expr r body))
```

**`Alpha_handler`.** Before:

```coq
  | Alpha_handler (r : renaming) (k : continuation) (params : list variable)
      (metadata : handler_metadata) (handler body : expr) :
    code_closed handler = true ->
    (∀ y : variable,
      y ∉ params -> var_free_in_expr y handler -> rename_variable r y = y) ->
    (∀ k' : continuation,
      cont_free_in_expr k' handler -> rename_continuation r k' = k') ->
    alpha_equivalent
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k
         (rename_handler r (Handler params metadata handler)) body)
```

After:

```coq
  | Alpha_handler (r : renaming) (k : continuation) (params : list variable)
      (metadata : handler_metadata) (handler body : expr) :
    code_closed handler = true ->
    fixes_names r
      (free_variables_of_handler (Handler params metadata handler))
      (free_continuations_of_expr handler) ->
    alpha_equivalent
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k
         (rename_handler r (Handler params metadata handler)) body)
```

**`Alpha_recursive_handler`.** Before:

```coq
  | Alpha_recursive_handler (r : renaming)
      (before : list (continuation * continuation_handler))
      (k : continuation) (params : list variable)
      (metadata : handler_metadata) (handler : expr)
      (after : list (continuation * continuation_handler)) (body : expr) :
    code_closed handler = true ->
    (∀ y : variable,
      y ∉ params -> var_free_in_expr y handler -> rename_variable r y = y) ->
    (∀ k' : continuation,
      cont_free_in_expr k' handler -> rename_continuation r k' = k') ->
    alpha_equivalent
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler) :: after) body)
      (Let_cont_recursive
         (before
          ++ (k, rename_handler r (Handler params metadata handler)) :: after)
         body)
```

After:

```coq
  | Alpha_recursive_handler (r : renaming)
      (before : list (continuation * continuation_handler))
      (k : continuation) (params : list variable)
      (metadata : handler_metadata) (handler : expr)
      (after : list (continuation * continuation_handler)) (body : expr) :
    code_closed handler = true ->
    fixes_names r
      (free_variables_of_handler (Handler params metadata handler))
      (free_continuations_of_expr handler) ->
    alpha_equivalent
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler) :: after) body)
      (Let_cont_recursive
         (before
          ++ (k, rename_handler r (Handler params metadata handler)) :: after)
         body)
```

The five statements `alpha_rename_canonical_statement`,
`alpha_let_canonical_statement`, `alpha_let_cont_canonical_statement`,
`alpha_handler_canonical_statement` and
`alpha_recursive_handler_canonical_statement` of
`FlambdaCanonicalFreeNamesSpec.v` said that the rules held in the
newer form. They are the rules now, and are deleted.

### 3.3 Two statements of `FlambdaAlphaSpec.v`

Each asks that the code is closed.

**The continuations of a recursive `Let_cont`.** Before,
`alpha_let_cont_recursive_statement`; now the lemma
`FlambdaAlphaProofs.alpha_let_cont_recursive`:

```coq
Definition alpha_let_cont_recursive_statement : Prop :=
  ∀ (r : renaming) (handlers : list (continuation * continuation_handler))
    (body : expr),
    code_closed (Let_cont_recursive handlers body) = true ->
    (∀ y : variable,
      var_free_in_expr y (Let_cont_recursive handlers body) ->
      rename_variable r y = y) ->
    (∀ k : continuation,
      cont_free_in_expr k (Let_cont_recursive handlers body) ->
      rename_continuation r k = k) ->
    alpha_equivalent (Let_cont_recursive handlers body)
      (Let_cont_recursive (rename_handlers r handlers) (rename_expr r body)).
```

After, `alpha_let_cont_recursive_canonical_statement`, in the same
file:

```coq
Definition alpha_let_cont_recursive_canonical_statement : Prop :=
  ∀ (r : renaming) (handlers : list (continuation * continuation_handler))
    (body : expr),
    code_closed (Let_cont_recursive handlers body) = true ->
    fixes_names r
      (free_variables_of_expr (Let_cont_recursive handlers body))
      (free_continuations_of_expr (Let_cont_recursive handlers body)) ->
    alpha_equivalent (Let_cont_recursive handlers body)
      (Let_cont_recursive (rename_handlers r handlers) (rename_expr r body)).
```

**Execution is equivariant in the variables.** Before,
`rename_variables_equivalent_statement`; now the lemma
`FlambdaAlphaProofs.rename_variables_equivalent`:

```coq
Definition rename_variables_equivalent_statement : Prop :=
  ∀ (r : renaming) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env),
    code_closed e = true ->
    (∀ k : continuation,
      cont_free_in_expr k e -> rename_continuation r k = k) ->
    equivalent (State e vars conts statics)
      (State (rename_expr r e) (rename_env r vars) conts statics).
```

After, `rename_variables_equivalent_canonical_statement`
(`FlambdaCanonicalFreeNamesSpec.v`):

```coq
Definition rename_variables_equivalent_canonical_statement : Prop :=
  ∀ (r : renaming) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env),
    code_closed e = true ->
    fixes (continuation_permutation r) (free_continuations_of_expr e) ->
    equivalent (State e vars conts statics)
      (State (rename_expr r e) (rename_env r vars) conts statics).
```

### 3.4 A fresh name is not free (`FlambdaFreshBindersSpec.v`, N2's)

Before, `fresh_not_free_statement`; now the lemma
`FlambdaFreshBindersProofs.fresh_not_free`:

```coq
Definition fresh_not_free_statement : Prop :=
  ∀ e : expr,
    code_closed e = true ->
    (И x, ~ var_free_in_expr x e) ∧ (И k, ~ cont_free_in_expr k e).
```

After: nothing new. It follows, on every term, from
`FlambdaFreshQuantifierSpec.for_fresh_avoids_statement`, which was
there, at the two lists of free names, and from
`var_free_among_statement` and `cont_free_among_statement`:

```coq
Definition for_fresh_avoids_statement : Prop :=
  ∀ names : list positive, И name, among name names = false.
```

### 3.5 What stays as it was

**A binding that is not used** (`FlambdaNominalSpec.v`). The three
statements, unchanged by that series. (Since then the first, and its
corollary below, have gained a premise for the heap: that the
defining expression is not a primitive of the heap,
`named_changes_heap defining_expr = false`. They are quoted here as
they were.)

```coq
Definition unused_let_statement : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    ~ var_free_in_expr x body ->
    expr_refines (Let x kind defining_expr body) body.

Definition unused_let_cont_statement : Prop :=
  ∀ (k : continuation) (handler : continuation_handler) (body : expr),
    ~ cont_free_in_expr k body ->
    expr_equivalent (Let_cont_non_recursive k handler body) body.

Definition unused_let_cont_recursive_statement : Prop :=
  ∀ (handlers : list (continuation * continuation_handler)) (body : expr),
    handlers_bind_statics handlers = false ->
    (∀ k : continuation,
      cont_free_in_expr k body -> lookup k handlers = None) ->
    expr_equivalent (Let_cont_recursive handlers body) body.
```

Their forms with the one function, which were and stay in
`FlambdaCanonicalFreeNamesSpec.v` as corollaries:

```coq
Definition unused_let_canonical_statement : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    among x (free_variables_of_expr body) = false ->
    expr_refines (Let x kind defining_expr body) body.

Definition unused_let_cont_canonical_statement : Prop :=
  ∀ (k : continuation) (handler : continuation_handler) (body : expr),
    among k (free_continuations_of_expr body) = false ->
    expr_equivalent (Let_cont_non_recursive k handler body) body.

Definition unused_let_cont_recursive_canonical_statement : Prop :=
  ∀ (handlers : list (continuation * continuation_handler)) (body : expr),
    handlers_bind_statics handlers = false ->
    (∀ k : continuation,
      among k (free_continuations_of_expr body) = true ->
      lookup k handlers = None) ->
    expr_equivalent (Let_cont_recursive handlers body) body.
```

**The relations themselves** (`FlambdaAlphaSpec.v`). The three
statements, unchanged:

```coq
Definition var_free_equivariant_statement : Prop :=
  ∀ (r : renaming) (x : variable) (e : expr),
    var_free_in_expr (rename_variable r x) (rename_expr r e)
    <-> var_free_in_expr x e.

Definition cont_free_equivariant_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (e : expr),
    cont_free_in_expr (rename_continuation r k) (rename_expr r e)
    <-> cont_free_in_expr k e.

Definition alpha_equivalent_free_names_statement : Prop :=
  ∀ e e' : expr,
    alpha_equivalent e e' ->
    (∀ x : variable, var_free_in_expr x e <-> var_free_in_expr x e')
    ∧ (∀ k : continuation,
        cont_free_in_expr k e <-> cont_free_in_expr k e').
```

The functions have their own, in the same file
(`free_variables_equivariant_statement`,
`free_continuations_equivariant_statement`,
`alpha_equivalent_free_variables_statement`,
`alpha_equivalent_free_continuations_statement`).

## 4. The lemmas that keep the older forms

Each has the name and the text that the statement had. The definition
of the statement is in the `Proofs` file, above the lemma, with a
comment that says it is no statement of a `Spec` file.

| Lemmas | In | Proofs that use them |
|--------|----|----------------------|
| `rename_let`, `freshen_let`, `rename_let_cont`, `rename_handler_params`, `rename_recursive_handler_params`, `rename_let_cont_recursive` | `FlambdaNominalProofs.v` | the newer forms; the soundness of `alpha_equivalent`; the binders at fresh names (N2) |
| `Alpha_rename_by_relation`, `Alpha_let_by_relation`, `Alpha_let_cont_by_relation`, `Alpha_handler_by_relation`, `Alpha_recursive_handler_by_relation` | `FlambdaAlphaProofs.v` | that `alpha_equivalent` is equivariant; the instances of the interface; the rules at fresh names, the recursive group, the converse of the comparison, inlining at fresh names (N2) |
| `alpha_equivalent_by_relation_ind` | `FlambdaAlphaProofs.v` | the eight inductions on a derivation (N1) |
| `alpha_let_cont_recursive` | `FlambdaAlphaProofs.v` | the recursive group (N2) |
| `rename_variables_equivalent` | `FlambdaAlphaProofs.v` | the newer form |
| `fresh_not_free` | `FlambdaFreshBindersProofs.v` | the binders at fresh names (N2) |

The rules in their older form, and the induction, are proved from the
rules and from the four statements of `FlambdaFreeNamesSpec.v`: from
the older premises to the newer where the code is closed, which each
rule asks; from the newer to the older whatever the code.

## 5. What changes in files of other owners

No statement of another owner changes but N2's one (3.4). No proof of
another owner changes but in these ten places.

- **N2.** `FlambdaFreshBindersSpec.v`: the statement of 3.4 goes; the
  header and three comments, which named statements of
  `FlambdaNominalSpec.v`, name `FlambdaCanonicalFreeNamesSpec.v`. In
  the proofs: the definition of that statement moves to
  `FlambdaFreshBindersProofs.v`; and ten uses of the five constructors
  take the name of the lemma with `_by_relation`
  (`FlambdaAlphaGroupProofs.v` two, `FlambdaCompareConverseProofs.v`
  three, `FlambdaFreshBindersProofs.v` three,
  `FlambdaInlineFreshProofs.v` two). Nothing else in those proofs
  changes.

No other file of another owner is touched.

A branch in flight: a proof that uses the older theorems by name
needs nothing. A proof that applies one of the five constructors with
the older premises takes the name with `_by_relation` at its rebase; a
proof by induction on a derivation that reads the premises in the
older form adds `using alpha_equivalent_by_relation_ind`.

## 6. Which form gives which

"Newer from older" is how the newer forms are proved
(`FlambdaCanonicalFreeNamesProofs.v`); "older from newer" is by the
lemmas `free_variables_fixed` and those beside it in
`FlambdaAlphaProofs.v`, which give the newer premise from the older
where the code is closed.

| Statements | Newer from older | Older from newer |
|------------|------------------|------------------|
| The six that rename (3.1), `alpha_let_cont_recursive`, `rename_variables_equivalent` (3.3): replaced | on every term | on every term: each asks that the code is closed |
| The five rules (3.2): replaced | the relation defined is the same; each older rule is a lemma | the induction in the older form is a lemma |
| `fresh_not_free` (3.4): deleted | | from `for_fresh_avoids_statement` and the statements that what is free by the relation is among the list, on every term |
| The three about a binding that is not used (3.5): kept | on every term (the corollaries) | only where the code of the body is closed |
| The three about the relations themselves (3.5): kept | | the statements about the functions do not give them where code is not closed |

Why the three about a binding that is not used say more in the older
form. They ask nothing of the code. The newer premise is that the
name is not among the free names, those in code included; the older,
that it is not free by the relation, which does not count a name that
occurs only in code. Such a binding may still be removed: the machine
runs code under its parameters and `my_closure` alone, so code never
reads a name that it does not bind. The rewrites of the validator
that remove bindings (`remove_unused_lets`, `remove_unused_conts`,
the sinking of continuations) test freedom without going into code
(`FlambdaNormaliseSpec.var_free_b`,
`FlambdaUnusedContsSpec.cont_free_b`), are stated of every term, and
are proved from the older forms. Of a well-formed term the two forms
say the same.

## 7. What is still written in the older style, and why

| What | Whose | Why |
|------|-------|-----|
| The three statements about a binding that is not used (`FlambdaNominalSpec.v`) | N1 | On purpose: section 6. They are the laws that the rewrites which remove bindings need, of every term |
| The three statements about the relations themselves (`FlambdaAlphaSpec.v`) | N1 | They say that the premise of the three above is kept by renaming and by alpha-equivalence. Where code is not closed the statements about the functions do not give them |
| The relations `var_free_in_expr`, `cont_free_in_expr` and those beside them, as definitions of `FlambdaNominalSpec.v` | N1 | For the six statements above, the rows below, and the proofs that use them |
| `FlambdaApplySpec.inlinable`, which says that the code inlined is closed and that the bindings do not capture | the integrator | It asks that the body of the code binds nothing static, so there is no code in it and the two accounts cannot differ. The definition is unfolded by proofs of the integrator, of N2 and of the types. To be restated at a quiet point |
| `FlambdaNormaliseSpec.expr_in_scope`, the check of scope in the test of inlining | the integrator | It is a check that runs, inside `inline_calls`, and is extracted. It agrees with `names_among` where code is closed (`FlambdaFreeNamesSpec.expr_in_scope_agrees_statement`). Replacing it changes the function, and replays every proof about `inline_calls` |
| `FlambdaWellFormedSpec.free_continuations`, by which a handler is found to be at toplevel and the exception handlers are checked | the integrator | A check that runs, a part of `well_formed_unit`, extracted. It agrees with the one function where code is closed (`free_continuations_agree_statement`), and `well_formed_unit` asks that it is |
| `var_free_b`, `cont_free_b`, the tests of freedom of the rewrites that remove bindings | the integrator (V4's passes use them) | They do not go into code, and need not: section 6. With the one function the rewrites would remove less of a term whose code is not closed and the same of a well-formed one |
| `var_free_in_simple` in the first model of the types (`rocq/types/naive/FlambdaTypesSpec.v`) | the types | It says that a variable is a given simple. There is no code and no binder in a simple |
| `var_free_in_simples` in two statements of `FlambdaPartialApplicationSpec.v`, written since the request was made | C3 | They say that a variable is not one of a list of simples, where there is no code and no binder |
| The suffix `_canonical`, and `FlambdaCanonicalFreeNamesSpec.v` as the place of the statements that rename | N1 | The names without the suffix are those of the older lemmas, which proofs use |

`rename_apart_equivariant_up_to_statement`
(`FlambdaRenameApartSpec.v`) was a row of this table when the request
was made. Its owner has restated it with the one function since, so
it is not one.

## 8. Going back, and going on

Going back: the definitions of the nine older statements are in the
`Proofs` files with their text unchanged, and the rules in their older
form are lemmas. Restoring them to the `Spec` files is a move, and no
proof changes.

Going on, each at a quiet point:

1. `inlinable` is restated, by or with its owner.
2. The two checks are replaced by `names_among` and
   `free_continuations_of_expr`, with the semantics tests run, since
   they are extracted. Their statements of equivariance
   (`FlambdaValidatorEquivarianceSpec.v`) then follow from those of
   the one function.
3. The proofs are moved off the older lemmas, a file at a time. When
   none uses them the lemmas go, the suffix is dropped, and the
   statements that rename go back to `FlambdaNominalSpec.v`. The
   relations stay, for the six statements that are written with them
   on purpose; `FlambdaFreeNamesSpec.v` then says how they relate to
   the functions and nothing else.

## 9. Validation

At the replay, from the root of the worktree: `make -C rocq -j8 all`
passes with 2113 claims, and `make -C rocq assumptions` passes. No
definition that is extracted changes, and the extracted files are as
they were after `make -C rocq all`, so the semantics tests were not
run again.

## 10. Findings

No statement was found false or empty in its older form. Two things
were found, neither a discrepancy with the compiler:

- the three statements about a binding that is not used say more in
  their older form (section 6). The first version of this series
  replaced them by the weaker forms and kept the older as lemmas; the
  reviewer and the orchestrator ruled that no statement gets weaker,
  and they are statements as they were;
- `rename_apart_equivariant_up_to_statement` said that the second
  renaming agrees with the first on the names that are free by the
  relations, and asked nothing of the code. So of a term whose code
  is not closed it let the two differ on a name that only code
  mentions. Its owner has restated it since: the two agree on the
  free names by the one function, those in code included.
