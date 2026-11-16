# C3: sets of closures. A core change request

From C3, for C0 (who applies it), R1 (who reviews the `Spec` diff and
section 4) and the human (section 7). Plan: `PARALLEL-PLAN.md`, sections
3, 6 and 7.

The change is a series of six commits made on the integration branch
at 0a2933e7c9, after C1's epoch, in the shape of an epoch (section 8).
Applied, it is b9cddb2863 to 312da818a2 on the integration branch. Every
commit of it builds, nothing is `Admitted`, and there is no new axiom
or `Parameter`. Section 9 says what was run.

This is the fourth version of the request. The first two were written
beside a prototype on the base before C1 (fbcbbade2c). What changed
since the third version, which was the series on f8c8224ee6:

- the static sets that one group binds may not have code in common
  (R2's finding R2-15; the orchestrator's decision under the human's
  ruling on code ids, which the human has confirmed): a new condition
  of `well_formed_unit` (sections 1 and 2), and three examples
  (section 8);
- what section 2 said of Simplify's new code ids was wrong, and is
  corrected: for static sets it makes them once for a group, not once
  for each set;
- the review comments for the human are seven, each at the definition
  that makes something undefined (section 1);
- symbols that are not bound are ruled on (operations on them are
  undefined, and well-formedness excludes them): the general condition
  is C0's to write, after this series (section 2);
- the invariant on values is C3's to prove after the series is applied
  (section 8).

What changed between the second version and the third:

- the change is rebased over C1's epoch: the forms of value and the
  readings of a closure are in `FlambdaValueSpec.v`, their statements in
  `FlambdaSemanticsSpec.v` (section 1);
- the human's rulings are in: two functions of a set may not have the
  same code id, which `well_formed_unit` now asks (section 2); a
  projection that names the wrong function slot stays undefined, with a
  review comment for the human in the `Spec` file, as has every other
  place where the change makes something undefined and that is not
  certainly right (section 7);
- what a renaming acts on in the new forms, and why a call through a
  closure of a set with the wrong number of arguments still has no step
  (section 2).

## The design in brief

Today a set of closures has one function, and a closure is its code id
and its value slots. The change adds constructors, and changes none:

- a closure of a set of several functions is a value of its own: the
  function slot it is at, the code id of every function slot of the set,
  and the values of the value slots;
- `closure_of`, which sees a value as "code and value slots", keeps its
  type and gives the code of the closure's own slot. So `call_function`
  and `step` are not touched: a call runs the code of the slot, with
  `my_closure` bound to the callee, which is the closure at that slot;
- a function reaches another of its set by projecting a function slot
  from `my_closure`. That is how a set is recursive; no value is cyclic;
- the closures of a static set are symbols. Each is bound to its closure,
  which names the symbols of the others; projecting a function slot from
  a symbol gives the symbol of that slot;
- `expr` and `static_binding` get no constructor. The compiler's `Let`
  that binds several closures, and its static binding of several
  symbols, are two derived forms defined in `FlambdaModel.v`
  (`let_closures`, `bind_closures`), with statements saying that they
  bind what they should;
- the terms can then write sets that are not well-formed Flambda 2: a
  closure at a slot its set has not got, a slot given twice, two
  functions of a set with the same code, a static set bound in part,
  two static sets of one group with the same code.
  The machine has a behaviour on each, which is a choice (section 2);
  `well_formed_unit` refuses them, but for a value slot given twice,
  and two statements say what the run of a well-formed unit keeps.

Today's forms stay and mean what they mean: a set of one function whose
function slot has no name. They are meant to go later (section 8).

## 1. The change to the `Spec` files

The diff is `git diff 58435c1de4 312da818a2 -- '*Spec.v' '*Model.v'`:
1272 lines added in 10 files, of which 478 are a new file. No existing
constructor and no existing statement is edited. One existing
definition is, apart from the cases added to matches:
`well_formed_unit` gains a conjunct (below).

### `FlambdaModel.v`

```
Definition function_slot : Type := positive.

Inductive named : Type :=
  ...
  | Closure_of_set (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * simple))
  | Project_value_slot_from (project_from : function_slot)
      (slot : value_slot) (closure : simple)
  | Project_function_slot (move_from move_to : function_slot)
      (closure : simple).

Inductive static_const : Type :=
  ...
  | Static_const_closure_of_set (slot : function_slot)
      (closure_symbols : list (function_slot * symbol))
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * simple)).
```

`Closure_of_set slot function_decls value_slots` is the closure at
`slot` of a new set: `function_decls` gives the code of each function
slot, in the order of `Function_declarations.funs_in_order`.
`Static_const_closure_of_set` is the same for a symbol; `closure_symbols`
are the symbols of all the closures of the set, this one included.

Two equality tests, beside the types they are about (C0 asked for them
here; the check of well-formedness below is what uses them):

```
Definition simple_eq_dec : forall s s' : simple, {s = s'} + {s <> s'}.
Definition static_const_eq_dec :
  forall const const' : static_const, {const = const'} + {const <> const'}.
```

Both are by `decide equality`. `FlambdaCompareSpec.v` has the like for
kinds and constants (`kind_eq_dec`, `const_eq_dec`), which could come
here too.

The two derived forms:

```
Fixpoint let_projections (base : variable) (base_slot : function_slot)
    (bound : list (variable * function_slot)) (body : expr) : expr :=
  match bound with
  | [] => body
  | (x, slot) :: bound =>
    Let x (Project_function_slot base_slot slot (Var base))
      (let_projections base base_slot bound body)
  end.

Definition let_closures (bound_vars : list variable)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (body : expr)
    : option expr :=
  match bound_vars, function_decls with
  | x :: bound_vars, (slot, _) :: other_decls =>
    if Nat.eqb (length bound_vars) (length other_decls)
    then
      Some
        (Let x (Closure_of_set slot function_decls value_slots)
           (let_projections x slot
              (combine bound_vars (map fst other_decls)) body))
    else None
  | _, _ => None
  end.

Definition bind_closures (closure_symbols : list (function_slot * symbol))
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) : list static_binding :=
  map
    (fun '(slot, s) =>
       Bind_symbol s
         (Static_const_closure_of_set slot closure_symbols function_decls
            value_slots))
    closure_symbols.
```

### `FlambdaValueSpec.v` and `FlambdaSemanticsSpec.v`

The forms of value, `closure_of` and the three new readings of a value
are in `FlambdaValueSpec.v`, where C1's epoch put the values. The cases
of `eval_named` and `eval_static_const`, and the seven statements, are
in `FlambdaSemanticsSpec.v`, which exports the value file. The matches
on `value` end with a wildcard, as the others of that file do.

```
Inductive value : Type :=
  ...
  | Value_closure_of_set (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * value)).

Inductive statically_allocated_value : Type :=
  ...
  | Static_closure_of_set (slot : function_slot)
      (closure_symbols : list (function_slot * symbol))
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * value)).
```

`closure_of` gains two cases, and keeps its type:

```
  | Value_closure_of_set slot function_decls value_slots =>
    let* code := lookup slot function_decls in
    Some (code, value_slots)
  | Value_symbol s =>
    match lookup s (static_values statics) with
    ...
    | Some (Static_closure_of_set slot _ function_decls value_slots) =>
      let* code := lookup slot function_decls in
      Some (code, value_slots)
```

Three new observations of a value:

```
Definition unnamed_function_slot : function_slot := 1%positive.

(* The slot the closure is at, the code of each function slot of its
   set, the values of the value slots. *)
Definition closure_in_set (statics : static_env) (v : value)
    : option
        (function_slot * list (function_slot * code_id)
         * list (value_slot * value)) :=
  match v with
  | Value_closure code value_slots =>
    Some (unnamed_function_slot, [(unnamed_function_slot, code)], value_slots)
  | Value_closure_of_set slot function_decls value_slots =>
    let* _ := lookup slot function_decls in
    Some (slot, function_decls, value_slots)
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_closure code value_slots) =>
      Some
        (unnamed_function_slot, [(unnamed_function_slot, code)], value_slots)
    | Some (Static_closure_of_set slot _ function_decls value_slots) =>
      let* _ := lookup slot function_decls in
      Some (slot, function_decls, value_slots)
    | _ => None
    end
  | Value_const _ => None
  end.

Definition project_value_slot (statics : static_env)
    (project_from : function_slot) (slot : value_slot) (v : value)
    : option value :=
  let* '(at_slot, _, value_slots) := closure_in_set statics v in
  if Pos.eqb at_slot project_from then lookup slot value_slots else None.

Definition project_function_slot (statics : static_env)
    (move_from move_to : function_slot) (v : value) : option value :=
  let unnamed :=
    if Pos.eqb move_from unnamed_function_slot
    then if Pos.eqb move_to unnamed_function_slot then Some v else None
    else None
  in
  match v with
  | Value_closure _ _ => unnamed
  | Value_closure_of_set slot function_decls value_slots =>
    if Pos.eqb slot move_from
    then
      let* _ := lookup slot function_decls in
      let* _ := lookup move_to function_decls in
      Some (Value_closure_of_set move_to function_decls value_slots)
    else None
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_closure _ _) => unnamed
    | Some (Static_closure_of_set slot closure_symbols function_decls _) =>
      if Pos.eqb slot move_from
      then
        let* _ := lookup slot function_decls in
        let* _ := lookup move_to function_decls in
        let* s' := lookup move_to closure_symbols in
        Some (Value_symbol s')
      else None
    | _ => None
    end
  | Value_const _ => None
  end.
```

`eval_named` gains three cases and `eval_static_const` one. A closure
is made only at a function slot of its set (the first line of the two
cases; section 7, decision 5):

```
  | Closure_of_set slot function_decls value_slots =>
    let* _ := lookup slot function_decls in
    let* vs := eval_value_slots vars value_slots in
    Some (Value_closure_of_set slot function_decls vs)
  | Project_value_slot_from project_from slot closure =>
    let* v := eval_simple vars closure in
    project_value_slot statics project_from slot v
  | Project_function_slot move_from move_to closure =>
    let* v := eval_simple vars closure in
    project_function_slot statics move_from move_to v

  | Static_const_closure_of_set slot closure_symbols function_decls
      value_slots =>
    let* _ := lookup slot function_decls in
    let* vs := eval_value_slots vars value_slots in
    Some (Static_closure_of_set slot closure_symbols function_decls vs)
```

`eval_static_bindings`, `call_function`, `step`, `state`, `behaves`,
`equivalent` and `refines` are not changed.

Seven statements are added, proved in `FlambdaSemanticsProofs.v`:

- `closure_in_set_of_statement`: `closure_of statics v = Some (code,
  value_slots)` if and only if `closure_in_set statics v = Some (slot,
  function_decls, value_slots)` with `lookup slot function_decls = Some
  code`, for some `slot` and `function_decls`. The two see the same
  closures.
- `closure_in_set_slot_statement`: if `closure_in_set statics v = Some
  (slot, function_decls, value_slots)` then `is_bound slot
  function_decls = true`. A closure is at one of the function slots of
  its set, with no premise: `closure_in_set` looks the slot up, for a
  value and for a symbol. T1's meaning of types uses it (the function
  type of a closure's own slot is that of its code), and asked for it
  under this name.
- `project_value_slot_defined_statement`: `project_value_slot statics
  project_from slot v = Some w` if and only if `v` is, by
  `closure_in_set`, at `project_from` and its value slots have `w` at
  `slot`.
- `project_function_slot_defined_statement`: if the projection has a
  value then `v` is, by `closure_in_set`, at `move_from` of a set whose
  `function_decls` have `move_from` and `move_to`. These are necessary
  conditions and not sufficient: from a symbol the projection also
  needs the set to name a symbol for `move_to`. No statement gives
  conditions under which the projection from a symbol has a value.
  What the value is, where there is one, is said by the next statement
  for a closure made by a `Let`, and by
  `project_function_slot_bound_statement` (below) for any closure,
  where the static sets are bound.
- `project_function_slot_of_set_statement`: from `Value_closure_of_set
  move_from function_decls value_slots`, where the set has both slots, it
  is `Value_closure_of_set move_to function_decls value_slots`.
- `closure_of_set_made_statement`: if `eval_named statics vars
  (Closure_of_set slot function_decls value_slots) = Some v` then the
  value slots have values `slot_values` and `closure_in_set statics v =
  Some (slot, function_decls, slot_values)`.
  `static_closure_of_set_made_statement`: the same of
  `eval_static_const` and a symbol bound to what it makes. So nothing
  that a term makes with the new forms fails to be a closure.

The header of `FlambdaSemanticsSpec.v` says which sets the compiler
makes and what the machine does with the others (section 2 has the same
text).

Seven review comments for the human (`CR mshinwell`), as asked for
wherever the model makes something undefined and it is not certain that
this is right (section 7), each at the definition that does so:

- in `FlambdaValueSpec.v`: at `closure_in_set` (a value of the shape of
  a closure at a slot its set has not got is not a closure; decision
  5); before `project_value_slot` and at `project_function_slot` (the
  slot named as the source is wrong; decision 3); at
  `project_function_slot` again (from a symbol, the symbol the set
  names is given, bound or not; decision 4);
- in `FlambdaSemanticsSpec.v`: at `eval_named` and at
  `eval_static_const` (a closure at a slot its set has not got is not
  made; decision 5); at `call_function` (a call of a closure whose code
  is not bound is undefined, while the closure is made).

### `FlambdaWellFormedSpec.v`

`well_formed_unit` gains a conjunct, `sets_well_formed e`:

```
Definition set_well_formed (slot : function_slot)
    (function_decls : list (function_slot * code_id)) : bool :=
  is_bound slot function_decls
  && distinct (map fst function_decls)
  && distinct (map snd function_decls).

Definition static_set_whole (bindings : list static_binding)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol s
      (Static_const_closure_of_set slot closure_symbols function_decls
         value_slots) =>
    set_well_formed slot function_decls
    && (if
          list_eq_dec Pos.eq_dec (map fst closure_symbols)
            (map fst function_decls)
        then true
        else false)
    && match lookup slot closure_symbols with
       | Some s' => Pos.eqb s s'
       | None => false
       end
    && forallb
         (fun '(slot', s') =>
            existsb
              (binds_symbol_to s'
                 (Static_const_closure_of_set slot' closure_symbols
                    function_decls value_slots))
              bindings)
         closure_symbols
  | ... => true
  end.

Definition static_set_code_ids (binding : static_binding) : list code_id :=
  match binding with
  | Bind_symbol _ (Static_const_set_of_closures code _) => [code]
  | Bind_symbol _ (Static_const_closure_of_set _ _ function_decls _) =>
    map snd function_decls
  | ... => []
  end.

Definition binds_symbol_of_set (binding binding' : static_binding) : bool :=
  match binding, binding' with
  | Bind_symbol _ (Static_const_closure_of_set _ closure_symbols _ _),
    Bind_symbol s' _ =>
    among s' (map snd closure_symbols)
  | _, _ => false
  end.

Fixpoint static_sets_code_disjoint (bindings : list static_binding) : bool :=
  match bindings with
  | [] => true
  | binding :: bindings =>
    forallb
      (fun binding' =>
         orb (binds_symbol_of_set binding binding')
           match
             inter (static_set_code_ids binding)
               (static_set_code_ids binding')
           with
           | [] => true
           | _ :: _ => false
           end)
      bindings
    && static_sets_code_disjoint bindings
  end.

Fixpoint sets_well_formed (e : expr) : bool :=
  match e with
  | Let _ d body => named_sets_well_formed d && sets_well_formed body
  | Let_static bindings body =>
    forallb (static_set_whole bindings) bindings
    && static_sets_code_disjoint bindings
    && forallb binding_sets_well_formed bindings
    && sets_well_formed body
  ...

Definition well_formed_unit (unit_exn : continuation) (e : expr) : bool :=
  statics_at_toplevel unit_exn true e
  && distinct (static_names code_ids_of_binding e)
  && distinct (static_names symbols_of_binding e)
  && sets_well_formed e.
```

`named_sets_well_formed` is `set_well_formed` of a `Closure_of_set` and
true of the rest; the recursion goes into handlers and into code. In
words: every closure of a set is at a function slot of its set, whose
function slots are distinct and whose functions have each their own
code id (the human's ruling; section 2); a group that binds a symbol to
a closure of a static set binds every symbol of that set, each to the
closure of the same set at its slot, the set having a symbol for each
of its functions, in their order; and the static sets that one group
binds have no code in common. The third is what the compiler's one
pattern binds (`Bound_static.Pattern.Set_of_closures`, against which
`Static_const.match_against_bound_static_pattern` checks the function
slots, in order).

The fourth (`static_sets_code_disjoint`) compares the bindings of a
group two by two. Two bindings pass if the second binds a symbol of the
set of the first (in a group whose sets are whole they are then
bindings of one set, which is counted once though each of its symbols
has a binding), or if their sets have no code id in common. A static
set of one function with no slot named (`Static_const_set_of_closures`)
is a set here like the others, so this part of the conjunct is not
true of every unit written with the old forms (section 4). Two static
sets bound by two groups may have the same code, and so may two sets
made by `Let`s: section 2 says why the rule stops at a group.

`binds_symbol_to` compares static constants, with
`static_const_eq_dec` of `FlambdaModel.v` (above).

The check is here, and not in C2's kind checker, on C0's condition: it
is about static bindings and not about kinds, and the human has said
that real units may fail a check of kinds, so a premise that only that
checker could discharge would be one that real programs do not meet.

No statement of the file is edited. `well_formed_never_faults_statement`
takes `well_formed_unit` as its premise, so it is now about fewer units:
section 4.

### A new file pair, `FlambdaClosuresSpec.v` and `FlambdaClosuresProofs.v`

C3's (S4's once it is staffed). Seventeen statements, all proved:

- `let_closures_runs_statement`: where the variables are distinct and the
  value slots have values, `let_closures` runs, in as many steps as it
  has variables, to its body with each variable bound to
  `Value_closure_of_set` at the slot of the function in its position.
  `let_closures_undefined_statement`: where a value slot has no value,
  its first step is undefined.
- `bind_closures_binds_statement`: evaluating `bind_closures` binds each
  symbol to `Static_closure_of_set` at its slot (`static_set_bound`).
  `static_set_bound_extend_statement`: that survives every extension of
  the static environment.
- `project_static_set_statement`: under `static_set_bound`, the
  projection of a function slot from a symbol of the set is the symbol of
  that slot, and that symbol is, by `closure_in_set`, the closure there.
- Three rewrites of terms, all equivalences, which are what Simplify does
  to a projection from a closure of known type. In the scope of
  `let_closures`, `Project_function_slot f g (Var x_f)` may be replaced by
  `Simple (Var x_g)` (`project_function_slot_let_closures_statement`).
  Under `Let_static (bind_closures ...)`, `Project_function_slot f g
  (Symbol s_f)` may be replaced by `Simple (Symbol s_g)`
  (`project_function_slot_static_statement`), and
  `Project_value_slot_from f w (Symbol s_f)` by `Simple contents`, where
  the set gives `contents` to `w` (`project_value_slot_static_statement`).
- `apply_sibling_statement`: `Let x (Project_function_slot f g closure)
  (Apply x Indirect ...)` is equivalent to the same with `Direct code`,
  where `closure` is a closure made by a `Let`, at `f`, of a set whose
  `g` has `code`.
- `closure_code_determines_slot_statement`: if `closure_in_set statics
  v = Some (slot, function_decls, value_slots)`, the code ids of
  `function_decls` are distinct, `closure_of statics v = Some (code,
  value_slots')` and `lookup slot' function_decls = Some code`, then
  `slot' = slot`. In a set whose functions have each their own code,
  as those of a well-formed unit have, the code a closure runs says
  which slot it is at. This is what the new condition of
  well-formedness lets a statement say; section 2 says what it does
  not give.
- `parity_let_statement` and `parity_static_statement`: two units in
  which `even` and `odd` call each other through `my_closure` and read
  their answer at 0 from a value slot, one with the set made by
  `let_closures` and one with `bind_closures`, exit with the right answer
  on 2 and on 3. (The example was written before the primitives were
  data, so the predecessor comes from a `Switch`.) They use the three
  new forms of `named` and show that the statements above are not
  about terms that are all undefined.
- About the run of a well-formed unit from the empty state, in the
  manner of `well_formed_never_faults_statement`:
  - `well_formed_makes_closures_statement`: no state of the run is one
    in which a closure is to be made at a slot its set has not got
    (`makes_closure_outside_set`, a test on a state). So a well-formed
    unit is never undefined for that reason.
  - `well_formed_sets_bound_statement`: in every state of the run, a
    symbol bound to `Static_closure_of_set slot closure_symbols
    function_decls slot_values` is the symbol that `closure_symbols` has
    for `slot`, and `static_set_bound` holds of that set
    (`static_sets_bound`). This is C0's condition: the premise of the
    statements about static sets holds wherever a well-formed unit goes.
- `project_function_slot_bound_statement`: where `static_sets_bound`
  holds, if `project_function_slot statics move_from move_to v = Some w`
  then `v` is, by `closure_in_set`, at `move_from` of a set, and `w` is
  the closure of the same set (the same `function_decls` and slot
  values) at `move_to`. It needs no premise for a closure made by a
  `Let`; from a symbol it is false without one. It is about the result
  of a projection that has a value, and does not say that there is one:
  `static_sets_bound` holds of a static environment in which a set
  declares two function slots and names a symbol for one of them only,
  and the projection to the other has no value there.
- `parity_well_formed_statement`: the two units of the example are
  well-formed, so the three statements above are about some unit with a
  set of two functions. `partial_static_set_statement`: a unit that
  binds one of the two symbols of the static set and projects the other
  exits with a symbol that is bound to nothing, and is not well-formed.
  This is R2's example, kept as a statement so that the choice of
  decision 4 can be seen.

### Cases added in other `Spec` files

Each of these definitions matches on `named`, `static_const`, `value` or
`statically_allocated_value`, and gains a case for each new constructor;
none changes on the old ones.

- `FlambdaNominalSpec.v`: `rename_named`, `rename_static_const`,
  `var_free_in_named`, `var_free_in_static_const`. The value slots and
  the closure projected from, as for today's forms.
- `FlambdaCompareSpec.v`: `named_eqb`, `static_const_eqb`. The same
  slots, the same code ids and the same symbols, in the same order
  (`names_eqb`, new).
- `FlambdaNormaliseSpec.v`: `named_in_scope`, `static_binding_in_scope`,
  `var_in_named`, `var_in_static_binding`, `named_may_go`, `fold_named`,
  `named_is_total`, `named_defined`, `symbols_of`. `named_defined` says
  that a `Closure_of_set` has a value where its value slots are in
  scope and its slot is among those of its set (`is_bound slot
  function_decls`), as `eval_named` now asks.
- `FlambdaUnusedStaticsSpec.v`: `value_avoids`, `named_avoids`,
  `static_const_avoids`, `allocated_free_of`.
- `rocq/types/naive/FlambdaAliasesSpec.v`: `canonical_named`,
  `canonical_static_const`, `typing_env_under_let`.

Two of them do more than repeat a case. `symbols_of` records that a
symbol bound by `Static_const_closure_of_set slot _ function_decls _` is
a closure of `lookup slot function_decls`, so `inline_known_calls` inlines
calls of closures of sets. And in `FlambdaUnusedStaticsSpec.v` every
code id of a set and every symbol of its closures count as referred to,
so a static set is kept or removed whole (section 4).

## 2. The model of a set of closures

**What a closure value is.** One value for each function slot of a set.
The closures of a set share `function_decls` and `value_slots` and
differ in `slot`. A closure does not hold its siblings: the closure at
another slot is computed (`project_function_slot`), so values stay
finite trees and `=` on values keeps its meaning.

**How the functions refer to each other and to their value slots.** By
projecting from `my_closure`, as in the compiler: `Project_function_slot
f g (Var my_closure)` for a sibling, `Project_value_slot_from f w (Var
my_closure)` for a value slot. The value slots are those of the set,
evaluated once where the set is made. For a set made by a `Let` they
cannot refer to the closures of the set: the variables bound to those
are not in scope where the value slots are evaluated, here or in the
compiler. For a static set they can: a value slot may be `Symbol s`
with `s` a symbol of the same set, since a symbol is a value whether or
not it is bound yet. That is a recursive reference, and the value is
still finite, because a symbol is a name and not the closure it is
bound to.

**`my_closure`.** `call_function` is as it was: it binds the
`my_closure` of the code called to the callee. The callee is a closure
at one slot, and the code called is the code of that slot
(`closure_of`). So each function of a set gets the closure at its own
slot, or the symbol of that closure if it was called through a symbol.
(Unless a parameter of the code has the name of its `my_closure`, which
then hides it: the parameters are bound after it. Well-formedness now
asks that the binders of code are distinct, so no code of a well-formed
unit has such a parameter; "Sets that share code" below.)

**The projections are constructs of `named`**, like today's
`Project_value_slot`, and not primitives: their meaning needs the static
environment and is fixed by that of closures. C1 keeps
`Project_value_slot` in `named` (section 3). Both new projections are
undefined unless the closure is at the function slot they name
(`move_from`, `project_from`). Simplify may rely on that, on one of its
two paths (`simplify/simplify_unary_primitive.ml`, 32-53 and 55-93).
It first asks the type of the closure for the result, giving only
`move_to` or the value slot (`types/provers.ml`, 1159-1210;
`Row_like_for_closures.get_closure` and `get_env_var`,
`types/grammar/type_grammar.ml`, 4015-4035): if the type knows it, that
is the answer, and the slot named as the source is not looked at. Only
otherwise does it meet the type with "a closure at this slot", and that
meet is what replaces the projection by `Invalid` when the slot named is
wrong. Both answers refine a projection that is undefined; an answer of
`Invalid` would not refine one that had a value. Nothing in the compiler
checks the slot named: `set_wrong_slot.fl` (section 8) is a term in
which it is wrong, and discrepancy C3-1 has the references.
The human's ruling (D3): "probably another case where the model should
probably produce Invalid, but I'm not sure and a CR should be left in
the Spec file". So it is undefined, and the two projections carry the
comment.
Today's `Project_value_slot slot closure`, which names no function slot,
is not changed; `Project_value_slot_from` is the compiler's primitive in
full.

**Calls, and what a GADT can produce.** The human's rule is that
whatever programs with GADTs and the layout `any` can produce must be
undefined in the model: a call whose arities are not those of the code
called is the case that matters here. The change neither added to what
the machine checks of a call nor took from it, because it did not
touch `Apply`, `call_function` or `step`: at that stage a call with as
many arguments as parameters had a step whatever their kinds. The
series on kinds has since made such a call undefined
(`requests/C2-kinds.md`): `call_function` compares the kinds of the
arguments with the arity of the call, and the arities of the call with
those of the code, and has no step where they differ. The test is made in
the same place and on the same code for a closure of a set: a call
reads its callee with `closure_of`, which for a closure of a set gives
the code of the closure's own slot. `step_apply_closure`
(`FlambdaSemanticsProofs.v`) is the one way in which a call has a
step. Likewise a projection from a value that is not a closure, which
such programs can also contain in branches that are never taken, is
undefined.

**Names.** For each new form the renaming and the free names are in the
same change, and agree (`FlambdaNominalSpec.v`: `rename_named`,
`rename_static_const`, `var_free_in_named`,
`var_free_in_static_const`; `FlambdaNominalProofs.v` and
`FlambdaAlphaProofs.v` have the cases). A renaming acts on the
variables only: those of the value slots, of the closure projected
from, and the variables that `let_closures` binds, each by a `Let`, so
their scope is that of a `Let`. No binder is added. Function slots and
value slots are not renamed, here or in the compiler; nor are the code
ids of the functions and the symbols of a static set, like every code
id and symbol of the model today (the compiler's `Bound_pattern.Static`
does not permute the names it binds). The header of `FlambdaModel.v`
says so. The statements compare terms with `expr_equivalent` only: the
change adds no notion of equality up to bound names. That the variables
of `let_closures` are distinct is a premise of
`let_closures_runs_statement`, as the compiler's pattern has distinct
variables.

**A set bound by `Let`.** The compiler binds one variable to each
closure (`Bound_pattern.Set_of_closures`). The model's `Let` binds one
variable, and stays so: `let_closures` is a `Let` of the first closure
followed by a `Let` of the projection of each other closure from the
first. A closure has no identity in the model, so this is the same
thing, and `let_closures_runs_statement` says so. It is one creation and
then projections, not one creation for each closure, so that it stays
right if closures ever get addresses (C4). The alternative, a
constructor of `expr` binding a list of variables, was rejected for its
cost: a case in every function and every proof by cases on `expr`, in
every file.

**A set bound by `Let_static`.** The compiler binds the symbols of a set
with one pattern (`Bound_static.Pattern.Set_of_closures`). Here each
symbol has its own `Bind_symbol`, which repeats the set and lists the
symbols of all its closures; `bind_closures` makes them. So everything
about static bindings (names bound once, toplevel, splitting a group,
removing unused ones) applies unchanged, a symbol at a time. The price:
the model does not force the symbols a closure names as its siblings to
be bound to those siblings. A term made by `bind_closures` binds them
so (`bind_closures_binds_statement`), and the converter makes no other;
`well_formed_unit` asks for it, and the run of a well-formed unit keeps
it (`well_formed_sets_bound_statement`; section 1). The alternative, a
constructor of `static_binding` binding several symbols, costs a case in
every function and proof about bindings (well-formedness, unused
statics, comparison, merging of groups), and would still not make an
arbitrary static environment consistent, which is what statements under
all environments quantify over. Which of the two it is to be is
decision 4 of section 7.

**What the compiler can make, and what the machine does with the
rest.** Because a closure is written one at a time, and a static set a
symbol at a time, the terms can express sets that the compiler does not
make. In a set the compiler makes, every closure is at a function slot
of the set, no function slot and no value slot is given twice, the
functions have each their own code id, the symbols of a static set
are bound together, each to the closure at its slot, and the static
sets of one group have no code in common. For the others
the machine has to do something, and what it does is a choice and not
a model of anything in the compiler:

- a closure at a function slot that is not among those of its set is
  not made: making one, by a `Let` or statically, is undefined. A value
  of that shape can still be in an environment that a statement
  quantifies over; it is not a closure (`closure_in_set` has no value on
  it), so calling it or projecting from it is undefined. Decision 5;
- where a function slot or a value slot is given twice, the first of
  the two is the one that a lookup finds, as everywhere in
  `FlambdaMapSpec.v`. The second is not removed and is not without
  effect: every value slot is evaluated where the set is made, so a
  second one for the same slot that mentions an unbound variable makes
  the allocation undefined, and the closure holds both lists as they
  were given, which is what an exit shows of it. A repeated slot alone
  is never a reason for being undefined. For a function slot given
  twice, the two views that the compiler has of the functions of the
  set would disagree, and Simplify fails on it (discrepancy C3-5). A
  value slot cannot be given twice in the compiler, whose value slots
  are a map;
- a set in which two function slots have the same code id is made, and
  each of its closures runs that code with `my_closure` bound to
  itself; so are two static sets of one group that have the same code.
  See "Sets that share code" below;
- a closure of code that is not bound is made, and calling it is
  undefined, as for today's closures. It cannot be undefined where it is
  made: a group may bind a symbol to a closure before the code;
- nothing checks, where a static closure is made or where it is
  projected from, that the symbols it names for its set are bound to
  the closures of that set. A projection gives the symbol named, bound
  or not. Decision 4.

`well_formed_unit` refuses a unit with a closure at a slot its set has
not got, with a function slot given twice, with two functions of a set
that have the same code id, with a static set that a group binds in
part, or with a group two of whose static sets have the same code:
five things. It does not look at value slots: a unit with a value
slot given twice is well-formed. The test tool runs it on the term
before and on the term after Simplify. What the theorems that do not
take `well_formed_unit` as a premise then say of such sets is in
section 4.

It does not yet refuse a unit that mentions a symbol or the code of a
closure that nothing binds. The human has ruled on symbols (D8 of
`DECISIONS.md`: an operation on a symbol that is not bound is undefined,
and well-formedness excludes such terms). The condition is about every
mention of a symbol, not only those of sets, so it is C0's: row 4 of
`requests/C0-well-formed.md`, written after this series, with the
statement that no state of the run of a well-formed unit reads a name
that is not bound. `well_formed_sets_bound_statement` is the part of it
about the symbols of static sets, and is there to be used. Whether the
code of every closure is to be bound by the time it is called is not
decided; the comment at `call_function` asks.

**Sets that share code.** The first rule: the functions of a set
of closures have each their own code id. A term in which two function
slots of one set have the same code id is not well-formed Flambda 2.
This is the human's ruling on discrepancy R2-7, where the compiler
builds such a term and Simplify fails on it ("The following code_id has
been bound twice": it gives each old code id of a set one new code
id): "that sounds like a problem with the model; this is not a
well-formed term". So the hole was in the model's well-formedness, and
`set_well_formed` now asks for it (`distinct (map snd function_decls)`),
of a set made by a `Let` and of a static set alike. The machine is not
changed: it still runs such a set, as it runs the other sets that are
not well-formed. The converter never makes one from a unit the compiler
accepts: it numbers code ids apart, but for a new version that it
identifies with its deleted old one, and those two are not in one set,
the old one being then used nowhere. `set_shared_code.fl` is R2's term;
the tool refuses it as not well-formed and does not run Simplify.

What the rule gives: in a set of a well-formed unit, the code of a
closure determines its slot (`closure_code_determines_slot_statement`).
So when code is called, the closure called is at the one slot at which
its set has that code, whichever closure of the set it is; and where
`my_closure` stands for the closure called, the projections that the
code makes from `my_closure`, which name that slot, do not name a wrong
one. `my_closure` stands for it unless a parameter of the code has the
same name: a call binds `my_closure` and then the parameters.
`well_formed_unit` asks that the binders of code are distinct
(`FlambdaBindersSpec.binders_distinct`; R2's finding), so in the run of
a well-formed unit it does stand for it
(`well_formed_call_binds_my_closure_statement`). As a fact about every
closure in a state of a run it is part of the invariant on values
(section 8).

What the rule does not give: the code of a closure does not determine
its set. Two sets of a unit may have the same code ids, with other
value slots, and the compiler accepts that
(`set_code_in_two_sets.fl`: two sets made of the same two pieces of
code, one after the other; Simplify inlines the calls of both, and the
unit is well-formed). So a rewrite of the body of code that uses what
is known of the value slots of `my_closure`, or of the symbols of its
siblings, still needs a premise saying which set the callee belongs to.
Simplify has one by construction, because it gives the code of the
sets it simplifies new code ids (discrepancy C3-3): for a set made by a
`Let`, one map from old code ids to new for that set; for static sets,
one map for all the sets of a group (below). That premise is S4's to
state (section 8).

The second rule: the static sets that one group binds have no code id
in common. This is R2's finding (R2-15). Simplify takes all the sets of
a group together (`simplify/simplify_static_const.ml`, 447-477, which
collects them, and `Simplify_set_of_closures.simplify_lifted_sets_of_closures`,
`simplify/simplify_set_of_closures.ml`, 1161-1210), and keeps one map
from old code ids to new for all of them
(`Simplify_set_of_closures_context.compute_old_to_new_code_ids_all_sets`,
`simplify/simplify_set_of_closures_context.ml`, 219-244). It then
simplifies the code of each function of each set and binds the result
to the new code id that the map gives the old one (`simplify_function0`,
`simplify/simplify_set_of_closures.ml`, 444-451). Two sets of the group
that have one code id therefore bind one new code id twice, and
`Flow_acc.record_code_id_binding` stops the compiler ("The following
code_id has been bound twice", `simplify/flow/flow_acc.ml`, 194-208).
Nothing checks it where the group is built. By the human's ruling on
R2-7 this is the same case: a term that the compiler cannot simplify is
not well-formed, and the hole was in the model. The orchestrator
extended the rule on code ids to it, the human has confirmed that
("correct"), and `sets_well_formed` asks for it
(`static_sets_code_disjoint`, section 1). The machine is not changed.

The rule stops at a group because Simplify's map does. Two static sets
bound by two groups, or two sets made by `Let`s, are simplified apart,
each with its own map, and may have the same code:
`set_static_separate_shared_code.fl` has the two sets of
`set_static_group_shared_code.fl` in two groups, the compiler accepts
it and the two runs agree. `set_static_group_one_function.fl` is the
same defect in the old form, two static closures of one function each,
with no function slot, bound by one group to the same code: the
condition covers it, and the compiler fails on it in the same way.

**Today's forms.** `Set_of_closures`, `Static_const_set_of_closures`,
`Value_closure`, `Static_closure` and `Project_value_slot` are a set of
one function whose slot has no name. They are read as the closure at
`unnamed_function_slot` of a set with that one function
(`closure_in_set`), so that a closure has one reading for terms and for
types (agreed with T1, section 3). `unnamed_function_slot` is the
number 1, and it can be observed: `Project_function_slot 1 1` has a
value on an unnamed closure, and `Project_value_slot_from 1 w` reads
its value slot. The number is not reserved: the converter may give 1 to
a function slot of the compiler, and a unit is converted with one
presentation throughout only because the converter does so (section
5). That nothing depends on the constant is a convention, which is one
more reason to retire the unnamed forms (section 8). The alternative,
to give today's
constructors a function slot and a list of functions in place, changes
the shape of every statement that mentions them and of every pass, all
in the epoch. Adding costs a period with two presentations (section 8).

**Left out**, as the header of `FlambdaModel.v` now says: a function slot
whose code is deleted (`Function_declarations.Deleted`; the converter
refuses it), the allocation mode (a set on the stack is not told from
one on the heap), `only_full_applications`, `Rec_info` and the depth
variable, and everything about the layout of a set in memory (offsets,
`Slot_offsets`).

## 3. How it meets C1, C2 and T1

Agreed by message with each; recorded in `status/C3.md`.

**C1 (primitives).** Landed (f8c8224ee6), and the series is on it. As
agreed: `Project_value_slot` stays in `named`, so the new projections
are there too; the value constructors are beside `Value_block` and
`Value_boxed_number`, in `FlambdaValueSpec.v`, with `closure_in_set` and
the two projections of values; their statements are in
`FlambdaSemanticsSpec.v`. Nothing of C1's matches on `Value_closure`:
closures are read through `closure_of`. C1's proofs by cases on a
value do not name the constructors; one counts them
(`eval_is_null_statics_extend` of `FlambdaPrimitiveProofs.v`, whose
pattern gets one bar more). C1's other files and `FlambdaValueProofs.v`
build with no edit. `eval_prim_avoids` is a
lemma, from the law `eval_prim_preserves`, which has no clause about
closures: with `value_avoids` extended to the new constructor its proof
is unchanged, and the build checks it (section 4).

**C2 (kinds).** `Let` will carry a kind; `Handler` and `Code` arities;
`apply_expr` two arities, which `call_function` compares with the
code's. C3 changes neither `code` nor `call_function`, so there is no
conflict of meaning. Over C2: `let_closures` and `let_projections` put
kind `Value` on each `Let`. In C2's files: a case of `kind_of_named` for
each of the three new forms (`Value` for a closure; for
`Project_value_slot_from`, the kind C2's table `slot_kinds` gives the
slot), a case of `kind_of_value` and of the well-kinded-value predicate
for `Value_closure_of_set` and `Static_closure_of_set` (their value
slots, as today's), and that a projection of a function slot keeps a
well-kinded closure well-kinded. C2's checker does not ask for code ids
or symbols in scope. The structural conditions on a set were to be a
conjunct of that checker; they are now in `well_formed_unit` (section
1), which does not depend on kinds. So nothing of this request waits
for C2: the C3 epoch can run before C2's or after it. If it runs
before, the kinds on the `Let`s of the derived forms and the cases in
C2's definitions are C2's to add in its own epoch.

**T1 (types).** `type_satisfied` takes the static environment and reads
a closure only through `closure_of`, `closure_in_set`,
`project_function_slot` and `lookup` in the value slots, never by
matching on a constructor. That is the compiler's `closures_entry`:
`function_types` are `lookup g function_decls`, `closure_types` hold of
`project_function_slot statics slot g v`, `value_slot_types` of `lookup
w value_slots`. T1 defines the observations with these names and
argument orders in its own file until the C3 epoch, which deletes T1's
copies. T1 asked that the unnamed forms be the closure at
`unnamed_function_slot` in the semantics too, and they are. Until then
T1's copies of `closure_in_set` and `project_function_slot` match on
`value` case by case, and stop compiling when the constructor is added;
the orchestrator has asked T1 to end those matches with a wildcard, and
the epoch checks it (section 5).

**What in this request depends on theirs.** On C1's: nothing more; it
has landed. On C2's: only the text of the diff. Whichever of the two
series is ready first goes next, and the other is rebased on it: over
C2 the `Let`s of the derived forms get the kind `Value` and C2's
definitions get the cases above; over C3, C2 adds them. No definition
or statement of this request depends on a choice C2 has still to make.

## 4. Which statements change meaning

**No statement is edited, and none is falsified.** Every one of the 137
statements of the integration branch is proved on the branch with its
text as it is.

**Every statement but one says more than it did, and no less.** They
quantify over terms, values and environments, and there are now more of
each. On terms and environments without the new constructors each
statement says exactly what it said: `eval_named`, `eval_static_const`
and `closure_of` are unchanged on the old constructors, and `step` is
unchanged.

**The one: `well_formed_never_faults_statement`.** Its premise is
`well_formed_unit`, which gains the conjunct `sets_well_formed` (the
five conditions of section 2). On a unit without the new constructors
four of the five are true (every case of `named_sets_well_formed` and
of `static_set_whole` on an old constructor is `true`). The fifth is
not: a unit written with the old forms in which one group binds two
symbols to static closures of the same code was well-formed, and is
not now (`set_static_group_one_function.fl`; the compiler fails on
it). So the statement says of every other unit without the new
constructors what it said, and says nothing any more of those. Of the
units with the new forms it speaks only of those whose sets are
well-formed. Its proof drops the conjunct and is otherwise as it was.
No example of the integration branch has such a group: all 27 print
what they printed.

**What the statements without that premise say of malformed sets.**
`expr_refines`, `expr_equivalent`, the congruences (`plug_refines`,
`plug_equivalent`) and every rewrite quantify over all environments. So
they cover values of the shape of a closure at a slot outside its set
(in an environment from the start; a term cannot make one with the new
forms), and static environments in which a set is bound in part. On
those they say what the machine does, which section 2 gives: such a
value is not a closure, and a projection from a symbol gives the symbol
named. Each of them is
proved of all that, so none can be wrong because of a malformed set; it
says more than is needed. What is proved of the run of a well-formed
unit is two things only: it does not make such a closure with the new
forms, and its static sets are bound. It is not proved that no such
value is in a state of that run. That is the invariant on values,
which is C3's to prove after the series is applied (section 8).
Two things follow for whoever states a rewrite:

- a statement that is true of well-made sets only takes a premise:
  `static_set_bound` for one set, `static_sets_bound` for all. Three
  statements do (`project_static_set_statement`,
  `static_set_bound_extend_statement`,
  `project_function_slot_bound_statement`); the rewrites of
  `FlambdaClosuresSpec.v` need none, because they have the binding of
  the set in sight. `well_formed_sets_bound_statement` gives the premise
  at every state of the run of a well-formed unit;
- where a term makes a closure outside its set, it is undefined. A
  refinement then holds whatever the other side is, and an equivalence
  needs the other side to be undefined too. That is why `named_defined`
  asks for the slot: `remove_unused_defined_lets_equivalent_statement` is
  an equivalence, and removing a `Let` that is undefined is not one.

The statements to look at, with what they now also say:

- **Inlining** (`FlambdaApplySpec.v`: `apply_indirect_known_statement`,
  `inline_statement`, `inline_let_code_statement`). Text and proofs
  unchanged (`FlambdaApplyProofs.v` builds as it is). Before: the callee
  is a closure of the code called, and the inlined body binds
  `my_closure` to it. After: the same, where the callee may be a closure
  of a set and "the code" is that of its slot; the projections that the
  inlined body makes from `my_closure` are projections from that
  closure, as in the call.
- **Static bindings** (`FlambdaStaticSpec.v`:
  `let_static_split_statement`, `project_static_closure_statement`,
  `let_static_code_refines_statement`; `FlambdaContextSpec.v`:
  `plug_refines_statement`, `plug_equivalent_statement`;
  `FlambdaWellFormedSpec.v`). Text unchanged; `FlambdaStaticProofs.v`
  builds as it is, `FlambdaWellFormedProofs.v` has one line more
  (above), and
  `FlambdaContextProofs.v` needs two lemmas (the projections read the
  static environment through its values only, which two related
  environments share). `let_static_split_statement` now also says that
  the bindings of the symbols of one set may be put in two groups; a
  projection in between gives a symbol that is not bound yet, which is a
  value, as any symbol is. The term with two groups is equivalent to
  the   other and is not well-formed: `sets_well_formed` asks for one
  group, as the compiler has one pattern. It is a sufficient condition
  for the sets to be bound, not a necessary one. In the other
  direction, the statement says that two groups, each binding a static
  set, are equivalent to one group binding both; where the two sets
  have the same code the two groups are well-formed and the one is
  not. The equivalence holds (the machine runs both); it is the
  well-formedness of the result that a pass which merges groups has to
  establish.
- **Unused statics** (`FlambdaUnusedStaticsSpec.v`:
  `drop_statics_refines_statement`,
  `drop_statics_program_refines_statement`). Text unchanged; the
  definitions under `expr_avoids` and `envs_free_of` gain cases. What is
  new in substance: a closure of a static set refers to the symbols of
  its siblings and to the code of every function of the set. So
  `drop_statics` refuses to remove one symbol of a set and keep another,
  and to remove the code of one function of a set that is kept. It has
  to: a projection from the symbol kept gives the symbol removed. The
  proof gains the invariant that what is projected from a value
  referring to no name removed refers to none.
- **The comparison** (`FlambdaCompareSpec.v`: `expr_eqb_sound_statement`).
  Text unchanged; `named_eqb` and `static_const_eqb` gain cases. Code
  ids must still be the same on the two sides, now those of every
  function of a set.
- **The validator** (`FlambdaValidateSpec.v`: `validate_unit_sound`).
  Text and proof unchanged.
- **`eval_prim_avoids`** (`FlambdaUnusedStaticsProofs.v`). On the base
  before C1 it was an axiom, and `value_avoids` gaining a case would
  have widened it. Since C1's epoch it is a lemma, proved from the law
  that a primitive makes its value out of its arguments
  (`eval_prim_preserves_statement`), which says nothing of closures. Its
  proof is not edited and holds with `value_avoids` extended to
  `Value_closure_of_set`: no primitive makes a closure, so none brings
  in the code of a set. Nothing goes to the ledger.

## 5. Fallout

Found by building. "Builds as it is" means no edit at all. The sizes are
those of the repair on the branch, lines added and removed.

Rocq, C0's files:

- `FlambdaModel.v`: the change, +132 −13 (the header).
- `FlambdaValueSpec.v`: the forms of value and the readings, +162 −10
  (the header). `FlambdaValueProofs.v` builds as it is.
- `FlambdaPrimitiveProofs.v`: +1 −1, one pattern.
- `FlambdaSemanticsSpec.v`: the cases and the statements,
  +186 −4 (the header).
- `FlambdaSemanticsProofs.v`: builds as it is. +241 −1 of new
  lemmas (section 8).
- `FlambdaWellFormedSpec.v`: the check of sets, +189 −5 (the
  header). `FlambdaWellFormedProofs.v`: +1 (the proof of
  `well_formed_never_faults` drops the new conjunct).
- `FlambdaContextProofs.v`: +28 −1. Two lemmas, two cases.
- `FlambdaExtract.v`: 2 lines of the list.
- `FlambdaNominalSpec.v` +17 −3, `FlambdaNominalProofs.v`
  +15 −4, `FlambdaAlphaProofs.v` +34 −18 (N1's pilot is
  in new files; these are as they were).
- Build as they are: `FlambdaMap*`, `FlambdaKindModel.v`,
  `FlambdaPrimitiveModel.v`, `FlambdaPrimitiveSpec.v`,
  `FlambdaContextSpec.v`, `FlambdaControlFlow*`,
  `FlambdaAlphaSpec.v`, `FlambdaApply*`, `FlambdaStatic*`.

Rocq, other owners:

- V1: `FlambdaCompareSpec.v` +21, `FlambdaCompareProofs.v`
  +83 −8, `FlambdaNormaliseSpec.v` +35 −10,
  `FlambdaNormaliseProofs.v` +48 −13,
  `FlambdaUnusedStaticsSpec.v` +24 −1,
  `FlambdaUnusedStaticsProofs.v` +199 −8.
  `FlambdaValidate*` build as they are.
- T3: `rocq/types/naive/FlambdaAliasesSpec.v` +12 −1,
  `FlambdaAliasesProofs.v` +14 −3. `rocq/types/naive/FlambdaTypes*`
  (T1) build as they are on the integration branch.
- T1, on files that are not on the integration branch yet: its copies
  of `closure_in_set` and `project_function_slot`, which match on
  `value` case by case, stop compiling unless they end with a wildcard
  (section 3). They are to be deleted when T1 rebases on this.
- C2, on files that are not on the integration branch yet: the cases
  of `kind_of_named`, `kind_of_value` and well-kinded values (section
  3).

Nearly all of it is of one kind: a proof that destructs a `named`, a
`static_const` or a `value` with a pattern naming every constructor
(there are about thirty) needs the new branches, and usually one more
line of the tactic already there. The repairs with content are in
`FlambdaUnusedStaticsProofs.v` (what a closure of a set, a value slot
and a function slot are, when projected from a value that refers to no
name removed) and `FlambdaNormaliseProofs.v` (`statics_know_symbols`
restated with `closure_of`).

OCaml:

- `middle_end/flambda2/tests/tools/flsemantics.ml` (V2, V3). Once the
  extracted files are regenerated it no longer compiles (its matches on
  the model's terms and values are not exhaustive, and warnings are
  errors), and `make boot-compiler` builds it: so the regenerated files
  and its repair go in one commit.
  On the branch: +176 −33. The converter learns sets of several
  functions (through the extracted `let_closures` and `bind_closures`),
  the two named projections and the numbering of function slots; the
  printer, `observe`, `unused_statics` and `code_id_used` gain cases
  (the tool has had no evaluator of primitives since C1's epoch). Where
  the unit before Simplify is not
  well-formed the tool says which part of well-formedness it lacks (the
  old message, or a new one for sets), and does not run Simplify. Where
  the unit after Simplify is not, the tool stops with an error ("the
  term after Simplify is NOT WELL-FORMED"), for the new conjunct as for
  the others. That is right as long as Simplify never puts the closures
  of one static set in two groups, which the compiler's single pattern
  ensures; the converter makes one group of each `Let` of the compiler
  and does not split a set either.
- What the converter still refuses of sets ("not checked: the model has
  not got ..."): a function slot whose code is deleted; a set of no
  function, which includes the empty static set that the compiler
  permits (`bound_identifiers/bound_static.ml`, 25; `terms/flambda.ml`,
  687-700; the empty binding by a `Let` the compiler itself rejects,
  `bound_identifiers/bound_pattern.ml`, 164; discrepancy R2-3); a `Let`
  whose variables are not as many as the functions of its set; a static
  set whose symbols are not at the function slots of the set, in order.
- The extraction list of `FlambdaExtract.v`: `bind_closures` and
  `let_closures`.
- `tests/semantics/extracted/`: regenerated by C0.
- Expected files of examples: none changes. On the branch the
  27 examples of today print what they printed, and eleven are
  added (section 8).

**How the converter chooses between the two presentations.** A unit all
of whose sets have one function, and which projects no function slot,
is converted as today, with the unnamed forms. Any other unit is
converted with the named forms throughout, and the two units of a check
are converted alike. So every example of today is converted to the term
it is converted to today, and its status cannot change. The switch is a
flag in the tool and is to go (section 8).

## 6. Trust

Nothing enters and nothing leaves the ledger's axioms: no axiom, no
`Parameter`, and `make -C rocq assumptions` passes on the branch with
`rocq/TRUST-LEDGER.md` untouched.

Two entries of "Trusted OCaml" have to be updated in the epoch, with
R1's sign-off. What is newly trusted of the OCaml, each a thing that no
statement covers:

- "The conversion of terms":
  - the order of the declarations: `function_decls` is the functions of
    the set in the order of `Function_declarations.funs_in_order`, and
    the variables of a `Let` are paired with them by position;
  - the correspondence of symbols: each closure symbol of the
    compiler's pattern is given with the function slot it is bound at.
    The OCaml checks that the function slots of the symbols are those
    of the set, in order, and refuses otherwise; `well_formed_unit`
    checks it again on the result;
  - the choice between the two presentations: one for a whole unit, the
    same for the two units of a check, the unnamed one only where every
    set has one function and no function slot is projected (section 5);
  - what the compiler's two patterns expand to is not in the OCaml: it
    is `let_closures` and `bind_closures`, in Rocq, reviewed and with
    statements.
- "The numbering of names" numbers function slots too: two function
  slots of the compiler get two numbers. A slot of the compiler may get
  the number 1, which is `unnamed_function_slot` (section 2).

`eval_prim_avoids` is not an entry any more: since C1's epoch it is a
lemma, and its proof needed no edit (section 4).

## 7. Decisions for the human

1. **What is observed of a closure at exit.** Today an exit shows its
   arguments, compared with `=`: of a closure made by a `Let`, its code
   id and its slot values; of a symbol, its name only. With this request
   a closure of a set shows its slot, the code id of every function of
   its set, and the slot values.
   - (a) Leave it so. Nothing in `behaves`, `refines` or
     `validate_unit_sound` changes. Two terms that return closures agree
     only if the whole sets do, code ids included; that is already so for
     one function (the note on code ids in `FlambdaCompareSpec.v`).
   - (b) Observe less of a closure (not its siblings, or nothing but that
     it is a function, as the test tool's own comparison does). That
     needs a relation on values at exit in place of `=`, which changes
     what `validate_unit_sound` says.
   - Recommendation: (a) now. (b) belongs with the two changes that must
     put a relation there anyway: the relation on code ids (S4) and
     observation up to the heap (C4).
2. **Identity of closures.** In the model a closure has no identity: two
   sets made alike are equal, and nothing can ask whether two closures
   are the same allocation. C1 has agreed that physical equality, when
   it is added, has no value on closures.
   - (a) Keep closures without identity.
   - (b) Give them addresses when the heap comes (C4).
   - Recommendation: (a), and tell C4's designer that `let_closures` is
     already one creation followed by projections, which is the shape
     (b) would need.
3. **A projection that names the wrong function slot is undefined.**
   `Project_function_slot move_from move_to v` and
   `Project_value_slot_from project_from w v` have no value unless `v` is
   a closure at `move_from` (`project_from`); the first also needs the
   set to have `move_to`.
   - (a) Undefined, as proposed. Simplify may turn such a projection
     into `Invalid` (when the type of the closure does not already give
     the result: section 2), which is a refinement only if the
     projection is undefined; where the type does give the result,
     Simplify answers with it, which refines an undefined projection
     too. And a model in which the projection had a value would accept
     a Simplify that changed `project_from` wrongly.
   - (b) Ignore the slot named, as today's `Project_value_slot` does,
     having no slot to name. Then the `Invalid` that Simplify can give
     is not a refinement.
   - Recommendation: (a). It is why `Project_value_slot_from` is added
     and `Project_value_slot` not reused.
   - The human, 2026-11-11: "D3 is probably another case where the
     model should probably produce Invalid, but I'm not sure and a CR
     should be left in the Spec file." So (a), with the comment at the
     two projections in `FlambdaValueSpec.v`. The same was asked "for
     the other cases like this, where it isn't clear that an Invalid is
     correct": decisions 4 and 5 below, and the closure of code that is
     not bound, have one each at the definition that makes the choice.
4. **What the machine does with a static set that is bound in part or
   inconsistently** (section 2, "A set bound by `Let_static`"). This
   decides the behaviour of the machine on terms that the compiler does
   not produce, and what every statement about static sets has to
   assume. `partial_static_set_statement` is the case in point: a unit
   binds one symbol of a set of two, projects the other and exits with
   a symbol that is bound to nothing.
   - (a) The machine does not check; well-formedness does. Each symbol
     has its own binding; projecting from a symbol gives the symbol
     named. `well_formed_unit` asks that a group binds a set whole, and
     `well_formed_sets_bound_statement` says that the run of a
     well-formed unit keeps every static set bound. Costs: every
     statement about a static set under all environments takes
     `static_set_bound` or `static_sets_bound` as a premise (three do
     today), and a reviewer has to look for a missing one;
     `well_formed_unit` has a conjunct more, with an equality test on
     static constants; a unit that is not well-formed can exit with an
     unbound symbol where the compiler's term could not exist.
   - (b) The machine checks at the projection: from a symbol it has a
     value only if the symbol named for `move_to` is bound to the
     closure at `move_to` of the same set. Costs: an equality test on
     values in `project_function_slot` (there is none; after C1 values
     have blocks and boxed numbers), so the definition T1's types read
     a closure through becomes harder to state facts about; the
     premise goes from the statements, and
     `project_function_slot_bound_statement` needs none. Simplify's
     rewrites stay refinements.
   - (c) One binding for a whole set: a constructor of `static_binding`
     that binds all the symbols at once. Costs: a case in every
     function and proof about bindings (well-formedness, unused
     statics, comparison, splitting and merging of groups, renaming),
     with new statements for each, where today's apply a symbol at a
     time; a larger epoch, by three or four days; and a static
     environment given to a statement could still be inconsistent,
     unless `static_env` changes shape too (sets kept apart, symbols
     pointing into them), which touches every statement that builds
     one.
   - Proposal and recommendation: (a). C0 accepts it on the condition,
     now met, that the check is in `well_formed_unit` and that the run
     of a well-formed unit is proved to keep the sets bound. The
     series is on (a).
5. **Making a closure at a function slot that its set has not got**
   (R2's finding; `DECISIONS.md` D20). A closure is written one at a
   time, so `Closure_of_set 1 [] []` is a term; the compiler cannot
   make such a closure. In the first version of this request it had a
   value, which nothing took for a closure and which a unit could
   return.
   - (a) Defined, and outside what the compiler produces. `eval_named`
     and `eval_static_const` make the value. "Outside" then has a
     precise meaning: `well_formed_unit` asks that every closure is at
     a slot of its set, and `well_formed_makes_closures_statement` says
     that the run of a well-formed unit never makes one that is not
     (the test `makes_closure_outside_set` is the same; it is then not
     a way of being undefined). Costs: no statement gains a premise and
     `named_defined` needs no condition; `closure_of_set_made_statement`
     and its static twin are false and go; a unit that is not
     well-formed exits with a value that is not a closure. Everything
     then rests on `well_formed_unit` being run: the converter makes
     such a term for no input (it goes through `let_closures` and
     `bind_closures`), and the tool checks the term before and the
     term after.
   - (b) Undefined where the closure would be made, when its slot is
     not among those of its set. Costs: `eval_named` and
     `eval_static_const` have one more way of having no value, so any
     statement that a `Closure_of_set` has a value takes the condition
     (`named_defined` does, and
     `remove_unused_defined_lets_equivalent_statement` keeps its text);
     as a refinement, removing an unused allocation needs nothing. In
     return what a term makes is always a closure
     (`closure_of_set_made_statement`, no premise). `well_formed_unit`
     has the same check as under (a), and
     `well_formed_makes_closures_statement` then says that a
     well-formed unit is never undefined for this reason. The converter
     and the tool have nothing more to guarantee.
   - (c) Undefined for every defect of a set: also a function slot or
     a code id given twice, and a static closure whose symbols are not
     bound to its siblings. A repeated slot or code id could be so (one
     more condition of the same kind; for a slot, against the rule of
     `FlambdaMapSpec.v` that the first binding counts). The static
     condition cannot be checked
     where the closure is made, since its siblings are bound after it
     in the same group: this is decision 4, (b) or (c).
   - Under each of these a closure of code that is not bound is made,
     and is undefined when called, as today: a group may bind the
     closure before the code.
   - Recommendation: (b), with the check in `well_formed_unit` as well.
     It is the plan's rule (undefined unless there is a reason), it
     costs one condition in one definition, and what it gives holds
     with no premise and does not rest on the converter. The first
     prototype was on (a), and the series is on (b), to show that (b)
     costs what is said here: the repairs were about a dozen lines of
     proof, in four files.
     Going back to (a) removes one line from each of two definitions
     and the two statements.
6. **Answered: two functions of a set may not have the same code id.**
   (Section 2, "Sets that share code"; discrepancy R2-7;
   D28 of `DECISIONS.md`.) The question was whether Simplify's failure
   on such a set is a missing check or a rule of Flambda 2. The human:
   "that sounds like a problem with the model; this is not a
   well-formed term". So it is a rule, and `well_formed_unit` asks for
   it; the tool says "not well-formed" before Simplify is run. The
   compiler does not check it where a set is made, and stops later with
   an error about a code id bound twice.
   **Extended to the static sets of one group** (R2-15): the compiler
   stops in the same way on a group two of whose static sets have the
   same code, and accepts the same two sets in two groups. The
   orchestrator decided under the ruling above that the sets of one
   group have no code in common, and that sets of different groups
   may; the human: "correct". `well_formed_unit` asks for it.

## 8. The series

The branch is six commits on the integration branch. Each of the first
five passes
`make -C rocq -j16 all` and `make -C rocq assumptions`; the last needs
the regenerated extracted files, and C0 applies it together with them,
because the compiler does not build in between. No step lemma of `step`
changes: `step` and `call_function` are not touched.

1. "sets of closures of several functions": `FlambdaModel.v`,
   `FlambdaValueSpec.v` and `FlambdaSemanticsSpec.v` with the diff of
   section 1; `FlambdaSemanticsProofs.v` with the new lemmas
   (`step_apply_closure`, the one way in which an `Apply` does not go
   wrong, with `my_closure` bound to the callee;
   `step_let_closure_of_set`, `step_let_project_value_slot_from`,
   `step_let_project_function_slot`) and the proofs of the seven
   statements; and every repair of section 5 (`FlambdaContextProofs.v`,
   `FlambdaNominal*`, `FlambdaAlphaProofs.v`, `FlambdaCompare*`,
   `FlambdaNormalise*`, `FlambdaUnusedStatics*`, `FlambdaAliases*`).
   They are one commit because the files do not build apart.
2. "well-formed sets of closures": `FlambdaWellFormedSpec.v` gains
   `sets_well_formed` (section 1), with the condition on the sets of
   one group, and `FlambdaWellFormedProofs.v` its one line.
3. "first facts about sets of closures": `FlambdaClosuresSpec.v` and
   `FlambdaClosuresProofs.v` (C3's).
4. This request, the status and C3's section of `OWNERS.md`.
5. "lines for the shared lists": the two files in `rocq/Makefile`, and
   `bind_closures` and `let_closures` in `FlambdaExtract.v`. From here
   the two new files are built and their statements checked.
6. The test tool and the eleven examples, the last.

For C0 and R1 after it: `rocq/HACKING.md` ("sets of closures of several
functions" leaves the list of what is not there; a paragraph on the two
presentations and the derived forms), and the two ledger entries
(section 6).

**Owed by C3 after it**, as leaf work in `FlambdaClosures*`, and not
part of this series: the invariant on values (R2). A value is valid
when every closure of a set in it is at
a function slot of its set, whose function slots are distinct and whose
functions have each their own code, and so on through the values of its
value slots and through blocks and boxed numbers. To prove: in every
state of the run of a well-formed unit, the values of the variables,
those of the environments that the continuations hold, the fields and
the value slots of what is statically allocated, and the arguments of
an exit are valid. It is kept apart from any condition that code or the
symbols referred to are bound. The case of a primitive is C1's law that
a primitive preserves a predicate on values; no axiom. It turns "a term
cannot make one" (section 4) into "a well-formed unit never holds one",
and with `closure_code_determines_slot_statement` it gives: in the run
of a well-formed unit, a closure that is called is at the slot of the
code called, and so is the `my_closure` of that code (the next
paragraph).

Done after the epoch, in `FlambdaClosuresSpec.v`: `value_valid` and
`allocated_valid` (what a symbol is bound to), and eight statements.
`well_formed_values_valid_statement` and
`well_formed_exit_valid_statement` are the invariant, over the one
traversal of where a state holds values
(`FlambdaSemanticsSpec.state_values_all`, the owner of names' work).
`well_formed_closure_slot_statement` says that a closure which the run
calls is at the only slot of its set that has the code called.
`call_binds_my_closure_statement` says that a call binds `my_closure`
to the closure called where `my_closure` is not also a parameter of
the code (R2's finding: a parameter of the same name hides
`my_closure`). That the binders of code are distinct is now a condition
of well-formedness, and two statements added since say what it gives:
the code that the run of a well-formed unit binds has distinct binders
(`well_formed_code_binders_statement`), and so a call in such a run
binds `my_closure` to the closure called with no premise on the code
(`well_formed_call_binds_my_closure_statement`).
`closure_in_set_valid_statement`,
`project_function_slot_valid_statement`, `eval_named_valid_statement`
and `eval_static_const_valid_statement` are what the invariant is made
of.

**The examples to add.** They are on the branch, with the real Simplify
run on each but the four that are not well-formed. Before the change
ten of the eleven are "not checked: the model has not got a set of
closures of several functions"; `set_static_group_one_function.fl`,
which has none, is held to be well-formed, and the tool dies in
Simplify. They are in the last commit.

- `set_mutual_recursion.fl`: `even` and `odd` in one static set, each
  reaching the other by `%project_function_slot` from `my_closure`.
  Simplify replaces each projection by the symbol of the sibling and
  makes the calls direct. The two runs are the same (59 steps before, 52
  after). Not validated: the term before would have to be rewritten, in
  the body of code, by what is known of `my_closure`, which is S4's work
  (`project_function_slot_static_statement` and `apply_sibling_statement`
  are the two facts it needs).
- `set_sibling_projection.fl`: a set made by a `Let`, with a value slot
  that both functions read, and a sibling projected at toplevel and
  called. Simplify inlines everything. The two runs are the same. Not
  validated: it needs closure types for the named forms in
  `follow_aliases`.
- `set_inline.fl`: two functions of one static set, called directly and
  inlined by Simplify, which then removes the set. Validated ("the same
  once normalised and unused static bindings removed"): Simplify changes
  the term, and the validator accepts the pair only because of the
  change. It goes through the new case of unused statics, a set removed
  whole.
- `set_wrong_slot.fl`: `%project_function_slot.[g].[f]` applied to the
  closure at `f` of a static set of `f` and `g`. The run before is
  undefined (decision 3). Simplify answers with the closure at `f`,
  from the type, without looking at the slot named, and the run after
  returns. "Anything goes, as the run before is undefined."
- `set_wrong_slot_invalid.fl`: the other of Simplify's two paths. A
  static set of `f`, `g` and `h`; the closure at `f` is hidden from the
  types by `%opaque`, a first projection `[f].[g]` tells them that it
  is at `f`, and a second, `[g].[h]`, names the wrong slot for it, to a
  slot that the set has. The type does not know the closure at `h`, so
  Simplify meets it with "a closure at `g`" and replaces the projection
  by `Invalid`. Both runs are undefined. With `[f].[h]` in its place
  the term is validated and both runs return, so it is the slot named
  as the source, and nothing else, that Simplify rejects.
- `set_duplicate_slot.fl`: a set in which two functions are given the
  same function slot. The compiler builds the term; the tool says "not
  checked: the term is not well-formed", and does not run Simplify,
  which fails on it (discrepancy C3-5).
- `set_shared_code.fl`: a set in which two function slots have the same
  code id (R2's term). Refused in the same way; Simplify fails on it
  (discrepancy R2-7).
- `set_code_in_two_sets.fl`: two sets, made one after the other of the
  same two pieces of code, with other value slots, and a function of
  each called. Well-formed; Simplify inlines both calls; the two runs
  are the same. Not validated. It is there to show that one code id
  may be in several sets.
- `set_static_group_shared_code.fl`: one group binds two static sets,
  each of two functions, made of the same two pieces of code (R2-15).
  Refused as not well-formed; Simplify stops on it ("code_id has been
  bound twice").
- `set_static_group_one_function.fl`: the same in the form of today:
  one group binds two symbols, each to a static closure of one
  function with no slot named, of the same code. Refused in the same
  way; Simplify stops in the same way.
- `set_static_separate_shared_code.fl`: the two sets of
  `set_static_group_shared_code.fl`, each bound by a group of its own.
  Well-formed; the compiler accepts it, and the two runs return the
  same block. Not validated. It shows that the rule stops at a group.

**After the epoch**, as leaf work and then one small request:

- V1 and T3 teach their passes the named forms where they know only the
  unnamed ones (`follow_aliases` and its closure types; the rest is done
  by the repairs above).
- V2 then makes the converter use the named forms for every unit, and
  deletes the switch. No example should change status; that is the test.
- A second request, "C3b", deletes the unnamed forms (`Set_of_closures`,
  `Static_const_set_of_closures`, `Project_value_slot`, `Value_closure`,
  `Static_closure`, `unnamed_function_slot`) and gives the named ones
  their names. It changes statements only by deleting cases and renaming
  constructors. It is an epoch, to be put on the list of epochs now.
  Before it can run:
  - the two steps above are done: every pass that knows the unnamed
    forms knows the named ones (`follow_aliases` and its closure types
    are what is missing today), and the converter makes named forms for
    every unit, with no example changing status;
  - T1's types read a closure only through `closure_of`,
    `closure_in_set` and the projections, with no case of their own for
    an unnamed closure;
  - no statement of an S stream is about `Set_of_closures` or
    `Project_value_slot` by name. Those that are (today
    `project_static_closure_statement` of `FlambdaStaticSpec.v`, and
    the closure type of `rocq/types/naive/FlambdaTypesSpec.v`, which says `v =
    Value_closure code slot_values`) are restated for the named forms
    in the request, which lists them;
  - C1's and C2's epochs have landed, since it renames constructors
    they match on.
  It needs no decision of the human unless a restated statement says
  something else than the old one.

**What could go wrong.**

- The rebase of one of C2 and C3 over the other. C2 adds an argument to
  `Let`, `Handler` and `Code`, so the patterns of the repairs move. The
  repairs are small and of one kind; the rebase over C1 took the
  resolution of nine files, of which two had content
  (`FlambdaUnusedStaticsProofs.v`, and the test tool, which C1's epoch
  had rewritten).
- A file written meanwhile that matches on `named` or on `value`
  constructor by constructor (the passes of the S streams, T1's
  denotation). `rocq/HACKING.md` now says to read a value through an
  observation and to end a match on `value` with a wildcard. T1's
  copies of the two observations are to end with one (section 3).
- The period in which the model has two presentations of a set. It is
  bounded by the steps above; if it drags on, passes get written for
  one presentation only.
- The static sets: a statement about a static set under all
  environments that forgets the premise `static_set_bound` is false. R1
  should look for that in what S4 writes. The premise can be
  discharged: `well_formed_sets_bound_statement`.

## 9. What was run, and what is left

Run on the tip of the series: `make -C rocq -j16 all` (161 statements
proved: the 137 of the integration branch and 24 new), `make -C rocq
assumptions`, `make -s boot-compiler`, the semantics tests (38
examples) and `SKIP_80CH=1 make -s fmt`. All pass. Each of the first
five commits was checked out and built by itself: `make -C rocq -j16
all` and `make -C rocq assumptions` pass at each. `make -s test` was
not run: the only OCaml changed is the test tool.

Proved, with nothing admitted:

- the `Spec` files compile with the diff, and so does everything else:
  every file of section 5 is repaired on the branch;
- the lemmas about the new forms (section 8, step 1) and the
  twenty-four new statements: seven in `FlambdaSemanticsSpec.v` and
  seventeen in `FlambdaClosuresSpec.v`;
- that the run of a well-formed unit keeps its static sets bound and
  makes no closure outside its set, by an invariant of `step` on the
  terms a state holds (its expression, the handlers of its
  continuations, the code bound) and on its static environment;
- of the theorems most at risk: the inlining theorems and the static
  binding theorems, whose proofs needed no change at all, so the claim
  that `closure_of` hides the change from calls is tested by the build;
  `plug_refines`, whose proof relates the static environments of two
  runs; and `drop_statics_refines`, the one whose invariant really
  changes (the symbols of siblings). I chose those because they are the
  proofs that reason about what a closure value is across two runs.

Left:

- The series is before C2's (section 3).
- `well_formed_unit` does not ask that the value slots of a set are
  distinct. The compiler's are a map, the converter makes them from it,
  and a lookup finds the first of two (section 2).
- The invariant on values (section 8): owed by C3, after the series is
  applied. Done since: section 8.
- That every symbol a unit mentions is bound (D8): C0's, after the
  series (section 2).
- No statement uses the condition on the static sets of one group. It
  is in `well_formed_unit` because the compiler stops on a group
  without it; what it gives a proof (among the static sets of one
  group, a code id is in one only) is for S4 to state when it needs
  it.
- No statement says when the projection of a function slot from a
  symbol has a value. In a well-formed unit it has one whenever both
  slots are among those of the set, since `sets_well_formed` asks that
  a static set names a symbol for each of its functions; that is not
  stated or proved. Done since, with the invariant on values:
  `project_function_slot_valid_statement`, whose premise holds in
  every state of the run of a well-formed unit.
- `follow_aliases` (`types/rocq/`) is repaired, not extended: it knows
  closure types for the unnamed forms only.
- The extracted files are regenerated on the branch and not committed,
  as the rules say; `flsemantics.ml` on the branch needs them.
