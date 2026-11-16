# T4: the meet of types: its contract and its functions

Second request of the meet stream. The first was the contract alone,
before any function; R1 reviewed it. This one has the contract as
revised after that review, the functions transcribed, statements about
them that are proved by running them, and what expanding the head of a
type keeps. It is for review by R1, and for T5 and T5b (the join), T2
(the environment) and T1 (the types) to check what is asked of them.
Section 14 says what changed since the first request.

Files, in `middle_end/flambda2/rocq/`:

- `FlambdaMeetSpec.v`: the contract. To be reviewed.
- `FlambdaMeetProofs.v`: its five closed statements, proved.
- `FlambdaMeetEnv.v`: the functions of `types/env/meet_env.ml`.
  `FlambdaExpandHead.v`: `Expanded_type` and `expand_head` of
  `types/expand_head.ml`. `FlambdaMeet.v`: the meet of
  `types/meet_and_n_way_join.ml`, `types/meet.ml`, and what
  `types/flambda2_types.ml` adds to `Typing_env`. Definitions only; to
  be reviewed against those files.
- `FlambdaMeetGrammar.v`: definitions that belong in the files of T1,
  T2 and C0 and wait for them there.
- `FlambdaMeetExamplesSpec.v`, `FlambdaMeetExamplesProofs.v`: eleven
  statements about the transcribed functions, nine proved by running
  them and two that are corollaries of the main theorem (section 15).
- `FlambdaMeetMatchExamplesSpec.v`,
  `FlambdaMeetProjectionExamplesSpec.v`,
  `FlambdaMeetTagExamplesSpec.v` and their proofs: what the meet
  answers on scenarios made after the calls that Simplify makes on
  small functions, twenty statements, proved by running it (section
  15).
- `FlambdaExpandHeadSpec.v`, `FlambdaExpandHeadProofs.v`: what
  `expand_head` keeps, eight statements, proved.
- `FlambdaMeetRenameSpec.v`: the meet commutes with a renaming of
  variables; nine statements, and six more for two joins that one
  renaming relates, proved (`FlambdaMeetRenameProofs.v`). To be
  reviewed (sections 16 and 17).
- `FlambdaMeetKeepsSpec.v`: what a meet keeps, for a join that has
  the half of its contract that is not about values; two definitions
  and seven statements, proved (`FlambdaMeetKeepsProofs.v`). To be
  reviewed (section 18). `FlambdaMeetValidity.v`: the device of the
  proof; definitions and a tactic, no statement.
- `FlambdaMeetParamsSpec.v`: the equations on parameters that the
  environment does not know; three definitions, an example and eight
  statements, proved (`FlambdaMeetParamsProofs.v`). To be reviewed
  (section 20).
- `FlambdaMeetChoicesSpec.v`: the three places where the compiler's
  meet is not the meet that is proved; three definitions, two
  examples and fourteen statements, proved
  (`FlambdaMeetChoicesProofs.v`). To be reviewed (section 21).

The compiler read: `types/meet_and_n_way_join.ml` (lines 1 to 1883, the
meet), `types/meet.ml`, `types/env/meet_env.ml`, `types/expand_head.ml`,
`types/flambda2_types.ml` (what calls meet), and of
`types/env/join_env.ml` what the meet's two calls reach
(`cut_and_n_way_join`, `import_var_transitive`,
`define_or_eliminate_variables_and_add_equations`).

## 1. The decision it is written under

One specification of meet; the traversal transcribed once, with the join
of scoped environments as a parameter; the n-way join first (D27). In
the compiler the meet proper calls the join at two places only,
`meet_disjunction` and `meet_row_like`, each time as
`Join_env.cut_and_n_way_join`. That call is the parameter.

## 2. The contract

A contract is a proposition about a function of the type that the
compiler's function has (`meet_function`, `add_equation_function`,
`meet_expanded_head`, `join_scoped_envs`). `None` is a fatal error of
the compiler, or a meet that ran out of fuel, or, at one place of the
transcribed functions, a state at which the model stops and the
compiler goes on (T4-10, section 12): there the compiler answers and
the model does not, whatever the fuel and whatever the choices of
section 11. What is to be proved of
the transcribed meet is `transcription_sound meet_to_prove`: for every
join that has `join_scoped_envs_sound` and every fuel, `meet_contract`.
It is proved (`meet_to_prove_sound`, `FlambdaMeetClosuresSpec.v`;
section 15).

`meet_sound`: if

- the inputs are well formed (`meet_inputs_well_formed tenv t1 t2`),
- the static environment is valid (`statics_valid statics`, of
  `FlambdaClosuresSpec.v`: among other things, a static set of
  closures names a symbol for each of its function slots),
- a valuation satisfies the environment
  (`typing_env_satisfied_strictly statics valuation tenv`),
- a value `v` has both types under it,
- and `meet tenv t1 t2 = Some result`,

then `result` is `Ok (t, tenv')`, and there is a valuation that has the
same value, or none, as the first for every variable that `tenv` knows
(`extended_from`), satisfies `tenv'`, and gives `v` the type `t`. The
agreement is exact, so that what the first valuation satisfied over
the variables that `tenv` knows, the second satisfies (T2's
`satisfied_but_coincide`). For a while it was weaker (a variable
without a value could be given one), to match the n-way join; R1
showed that the meet of an alternative of a disjunction could then not
give the join a valuation that still satisfies the environment of the
whole meet. The join is asked for the exact agreement too; what the
compiler's join has of it is in section 5.

So: the values of both inputs are kept; one valuation accounts for all
the equations and definitions returned; bottom is answered only when no
value has both types (`meet_bottom_sound`, which follows:
`meet_answers_bottom_statement`, proved).

The static environment is supposed valid in every statement that says
that no value is lost (this one, `meet_bottom_sound`, the three about
`Typing_env`, `meet_expanded_head_sound`, J1 of the join). Without it
`meet_sound` is false (R1): a static set of closures that names no
symbol for one of its function slots has every closures type at that
slot, by the meaning of closures types as T1 has it (what is asked of
the function at a slot is asked of the symbol that the set names for
it, and there is none), and the meet of two closure types of two
codes at that slot is bottom. `meet_closures_invalid_statics`
(`FlambdaMeetExamplesSpec.v`) is that case, proved. The other repair
would be for the meaning to ask that the symbol be there; T1 keeps
its meaning, and there is a `CR mshinwell` at `meet_sound`.

With it: `meet_keeps_well_formed` (the answer may be met again),
`meet_kinds`, `meet_bottom_env`; for `Typing_env.add_equation`,
`add_equation_sound`; and for `Typing_env.add_env_extension` and
`add_equations_on_params`, `add_env_extension_sound` and
`add_equations_on_params_sound`, each with the premise that every name
concerned has a value of the type given for it.

Not claimed: that the type answered has no more values than each input
(section 9).

## 3. What satisfies an environment, and poison (D26)

The contracts use T2's `typing_env_satisfied_strictly`
(`FlambdaTypingEnvSpec.v`): the environment is not bottom; every name
that has a value in the valuation has the type the environment gives
it, poison or not; aliases have the same value. A name may have no
value. They do not use `typing_env_satisfied`, which exempts the names
that poison touches.

Why, agreed with T2 and T5:

- **The exemption for poison is not enough, and a premise on one name is
  not either.** A meet reads the types of names (the target of an
  alias, the scrutinee of a relation, the names of an extension). If
  such a name holds poison and its type is not true of poison, the meet
  deduces from a false type. `x` holds poison, the environment has
  `x : null`: the value poison has the types `= x` and
  `any_non_null_value`, and their meet is bottom. With `x` and `y`
  holding poison, `x : Variant (is_int = z)`,
  `y : Variant (is_int = z')`, `z` holding 1 and `z'` holding 0, the
  meet of `= x` with `= y`, both of which poison has, makes `z` and
  `z'` aliases. The name can be reached through a field,
  so "the value of the name is not poison" for the name of
  `add_equation` does not cover it.
- **Strictness of every name does not survive the n-way join unless a
  name may have no value.** T5's example: the join defines in its result
  a variable `a` taken from one input, with the type it has there; under
  a valuation of another input no value need have that type. With
  partial valuations `a` has no value there. A type that mentions `a` in
  a live place has no value either, so the meet never reads the type of
  a name with no value for a value it has in hand.

- **A relation on a scrutinee that holds poison** (found by R1; D26 as
  revised). `p : unknown` holding poison and `t : Get_tag of p` holding
  0 was a strictly satisfied environment while T1's `relation_holds`
  related poison to every number; the meet of `= t` with the number 0
  gives `p` a type of blocks, which poison has not (for `Is_int`,
  `any_block`; and `Relation.of_const` reads `Is_int` of a poison
  constant as 1). Ruled: `Is_int` and `Get_tag` are not true of a
  scrutinee that holds poison, as in the core, where the primitives
  have no value on poison; T1 changes `relation_holds`; T2's
  `typing_env_satisfied` exempts a name whose type is a relation on a
  name that holds poison (done). The contract has no premise for it.
  T1's change is under this branch. `Is_null` keeps its
  clause (it relates poison to 0): the n-way join needs it, and what
  meet deduces from 0, `any_non_null_value`, poison has. T4-5.
- **A name with no value whose type is a relation on a name that holds
  poison** (R3) is strictly satisfied and harmless to meet: a meet
  makes a deduction only on behalf of a value in hand, and what tells
  it that a name has a value is the type that this value has (an
  alias, a relation, an equation of an extension). The functions that
  have no value in hand take the premise that the name has one.

The premise is therefore the strictness of the environment. It is the
compiler's unchecked assumption, and there is a `CR mshinwell` at
`add_equation_sound`. `add_equation_no_poison` (proved from the
contract) is `add_equation` for `typing_env_satisfied` where no name
holds poison. `meet_through_poison` (proved by running the meet) is
the first example above.

What the contracts do not cover: a handler in which unboxing has given
types to parameters that hold poison on the path taken. There the
environment is not satisfied strictly. I suggested to T2 that the
machine's valuation may leave such parameters without a value, which
would put those environments back under the contracts. T2 finds that
D26 implies it, and that what must be left without a value is more than
the poison (the `Is_int` and tag parameters that hold 0, and any value
built from the poison), so that a rewrite then needs "this variable has
a value" as a premise about the path. T2 keeps both notions for now and
has passed the alternative to the orchestrator (question 2 below).

## 4. Well-formedness of the inputs

`meet_inputs_well_formed tenv t1 t2`: the environment fits, each type
fits it, and the two types have one kind.

- A type fits an environment (`type_fits`) if it is well formed
  (`type_well_formed`), the environment knows its variables, and it is
  kinded there. That is two conditions. `type_kinded`: each alias is
  at the kind that the environment has for its target, and the
  variables for `Is_int`, `Get_tag` and `Is_null` are naked
  immediates. Meet does not need these kinds (it stops with a fatal
  error where they are wrong); the n-way join does, of the
  environments it is given, so meet takes them and gives them back.
  `type_equations_kinded`: each name that has an equation in an
  extension inside the type is of the kind of the type of its
  equation. Meet needs these itself, to keep the environment kinded:
  where it adds the extension of a case, the kind of a name is
  checked only where the type is not an alias and the canonical
  element of the name is a name. Both are premises about kinds, with
  a `CR mshinwell` (D10, D12, D32; T5-9). The traversals
  `aliases_with_kinds` (T5's) and `equation_names_with_kinds` are in
  T1's file; `type_kinded` and `type_equations_kinded` are T2's.
- An environment fits (`typing_env_fits`) if it has the invariant of
  typing environments (`typing_env_invariant`, T2's), its types are
  well formed (`types_well_formed`, T2's), it is kinded
  (`typing_env_kinded`, T2's: its types are kinded in it, two simples
  that it knows to be equal are of one kind, and a symbol that has a
  type has one of kind `Value`), the names of the equations inside
  its types are of the kinds of their types (`types_equations_kinded`,
  T2's, beside the last), and no name has an obviously bottom type
  in it unless it is bottom (`no_bottom_types`). Each of its types
  then fits it (`fits_types`, proved). The join owes all five of its
  result (J4, section 5). That a symbol's type is of kind `Value` is
  new with the move to T2's predicate: `Meet_env.add_alias` takes a
  symbol to be of that kind.
  `no_bottom_types` is there for the n-way join, which passes
  over an obviously bottom type and can then make the name an alias of
  what it is in the other environments joined, giving a value to a
  name that had none. The compiler's join needs it and it is not
  enough for that join (section 5). The model is stricter than the
  compiler here (the binary join makes such environments with
  `Meet_env.add_env_extension_maybe_bottom`); meet keeps the
  condition; there is a `CR mshinwell` (T5-10).

`type_well_formed : flambda_type -> bool` is T1's
(`FlambdaTypeWellFormedSpec.v`). It was a parameter of the contracts
until that file was under this branch, and is not any more: the
contracts are about T1's predicate. The condition on the environment
is T2's `typing_env_invariant` (`FlambdaTypingEnvInvariantSpec.v`); it
was a parameter too, until T2 had stated that the invariant is kept by
`Typing_env.add_alias` and `Typing_env.replace_equation` each on its
own, which are the two steps of `record_demotion`. What meet needs of
`type_well_formed`, agreed with T1:

- (a) the product of a case of blocks is no wider than the size of its
  index (`meet_mapping` makes a product with an empty component empty);
- (b) the function types, closure types and value slot types of a case
  of closures are of slots among those of its index, and an exact index
  has the slot of the case among its closures;
- (c) shapes are written in one way (`meet_row_like_for_blocks` compares
  them with `equal`);
- (d) kinds: the immediates of a variant, the contents of a boxed
  number, the closure types;
- (e) a set of numbers that is a head is not empty (`set_meet` answers
  an input unchanged when one set is included in the other). Here the
  model is stricter than the compiler (D32); T1 has it as a condition.

That a type is not an alias to a poison constant was asked in the
first request and is not any more (R1): the compiler makes such types
(`Typing_env.type_simple_in_term_exn`), and the revised meaning of
relations makes them harmless.

The n-way join does not keep (b) (T1-4). So the types that it answers
are not all inputs that this contract covers: see T4-3.

## 5. The join that meet is given

Type: `join_scoped_envs`, that is
`meet_expanded_head -> Scope.t -> Meet_env.t -> Typing_env.t ->
list Typing_env.t -> option Meet_env.t`: the compiler's
`cut_and_n_way_join ~meet_expanded_head ~cut_after initial_env
initial_tenv scoped_envs`, with `n_way_join_type` inside it. Agreed with
T5, whose own functions take a fuel that does not show here.

Contract (`join_scoped_envs_sound`), for a call as meet makes it
(`initial_tenv` is the typing environment of `initial_env`, `cut_after`
its scope, every scoped environment made from it by adding definitions
and equations: `scoped_env_of`), supposing that the
`meet_expanded_head` given has its contract, which is in two parts:
`meet_expanded_head_sound` (what it says of values) and
`meet_expanded_head_keeps` (the environment answered is the first
extended, it fits, the type answered fits it and is of the kind of
the inputs, and the variables that it has defined are new; no value
and no valuation is supposed), and supposing that no two of the
scoped environments have one simple at two kinds
(`scoped_envs_agree`):

- J1: a valuation that satisfies `initial_tenv` and one of the scoped
  environments extends to one that satisfies the result and agrees
  with it on every variable that this scoped environment knows;
- J2: the result is `initial_env` extended (`meet_env_extends`): same
  scope, earlier levels and `adding_equations_for_names`; every name it
  knew still has a type of the same kind, binding time and name mode;
- J3: a variable of the result that no environment given knows is not
  in `Meet_env.names_to_avoid initial_env`, and is in that of the
  result;
- J4: the result fits;
- J5: a simple that the result knows is, in the result, of the kind
  that a scoped environment has it at;
- J6: a variable of the result that `initial_tenv` does not know may
  not occur in terms (`Typing_env.mem` with the normal name mode for
  the least is false of it in the result), whether or not a scoped
  environment knows it.

J5 and the premise `scoped_envs_agree` are new, and so is the last
clause of `meet_expanded_head_keeps` (`new_names_avoided`: a variable
that the environment answered knows and the first does not was not
among the names to avoid of the first, and is among those of the
environment answered). After the join, `meet_disjunction` defines in
the result the variables of the two alternatives that the result does
not know, and answers types that fit the environments of the
alternatives. For the result to stay well formed and the types to fit
it, the variables of the two alternatives are to be different (the
second alternative is given the names to avoid of the first, so
`new_names_avoided` gives it, and with it the premise), and what the
join kept of them is to be at the kinds of the alternatives (J5).
Agreed with T5, which needs `new_names_avoided` of the meet for J3.
`forgetful_join_sound` is proved of the contract as it now is.

J6 was added on 2026-11-12, after the contract was signed, with a
third conjunct of `new_names_avoided` (a variable that a meet has
defined may not occur in terms), by T5b for T1's rewrite of the
`Switch`. With them a meet adds no variable that may occur in terms
(`meet_to_prove_keeps_terms`, `FlambdaMeetTermsSpec.v`); nothing in
the contract said the name mode of a new variable before. The meet
defines its variables in the name mode of the variables that exist
only in types (`define_in_types`), and so does the compiler's join
(`define_or_eliminate_variables_and_add_equations` of
`types/env/join_env.ml`). It asks one thing more of the join;
`forgetful_join_sound` is proved of the contract with J6.

J1 has one more conclusion since the second review, and one more
premise: no equation is added on a name that has no value
(`keeps_equations_valued`), given that the equations that the scoped
environment has after the scope of the meet are on names that have a
value (`equations_valued`). The meet makes extensions of the levels of
its scoped environments and puts them in the type that it answers; an
extension is true where each of its names has a value of the type of
its equation (T1's meaning, which a later meet relies on when it adds
the extension of a case and drops the case if that is bottom). The
same clause is a conclusion of `meet_sound`,
`meet_expanded_head_sound` and the `add_*_sound` contracts.

J1 is about one valuation and both environments. The first request
had instead, in `scoped_env_of`, that whatever satisfies a scoped
environment satisfies `initial_tenv`. R1 showed that meet cannot give
that: adding an equation can forget. With
`x : Value (unknown, may be null, is_null = z)`, adding
`x : any_non_null_value` answers `x : any_non_null_value` and says
nothing of `z` any more (`meet_disjunction`, where one alternative is
bottom); R1 ran it on both of the compiler's meets, and
`add_equation_forgets` proves it of the transcribed function.
`scoped_env_of` now says that the scoped environment fits and has
every name of `initial_tenv` with its binding time and a type of the
same kind. T5 has made the matching change (its premise on the source
is per valuation), and asks the scoped environments to be kinded,
which `scoped_env_of` gives. Since 2026-11-12 `scoped_env_of` also
says that the scoped environment knows no symbol that `initial_tenv`
does not (asked by T5: its join copies an alias to a symbol into a
type of the target; a meet defines no symbol, so the meet gives it).
The join needs the environment of the
meet to be satisfied by that valuation: for a name that every
environment joined has an equation on, it answers the meet of the
joined type with the type that `initial_tenv` records. The meet
supplies it by exact agreement: the valuation that an alternative
answers agrees with the one before on what `initial_tenv` knows.
`forgetful_join_sound` (proved) shows that the contract can be had:
the join that answers `initial_env` has it.

Found by T5 (T5-10), and decided: the compiler's n-way join has J1
only of the runs that have the values that it reads. It takes a
variable that an environment has a type for to have a value there;
a valuation that satisfies an environment strictly may give it none.
First, where the type joined for
a name `x` is a variant with a variable `r` for `Is_int`, it gives `r`
the type "`Is_int` of `x`" (`recover_inverse_relations`), also where
`x` has no value in the run (it was imported by an earlier join from
other uses) and `r` has one. Second, the join of the types that a name
has at the uses passes over a type that it finds to have no value (an
obviously bottom type, or a head of naked immediates that its
relations reduce to nothing), and the name then gets what it is at
the other uses, also on a run through the use at which it has no
value. `no_bottom_types` rules out the obviously bottom type only: R1
and T7 ran a stored type "0 or 2, and `Is_int` of `p`" with `p` a
tagged immediate at one use and the constant 1 at the other, and both
joins of the compiler answer the constant 1.
Weakening the conclusion of J1 (the names
whose type is a relation on a name with no value exempt) does not work
for the meet that follows: it reads the relation on behalf of the
value of `r` and adds an equation to `x`, whose stored type nothing
constrains when it has no value, and the meet of the two can be
bottom. A premise that the valuation gives every known variable a
value cannot be supplied after a join that imports from a subset. So
J1 stays strict, as the hypothesis under which the meet is to be
proved; T5 states the three conditions on a run under which the
compiler's join has what J1 says of the environment
(`scrutinees_have_values`, `joined_variables_have_values`,
`imported_variables_have_values`, in `FlambdaJoinSpec.v`). What J1
says of equations (`keeps_equations_valued`) needs a fourth: every
variable that has an equation in the level that the join adds has a
value at the use that the run goes through. The compiler's join gives
an equation to a variable that it makes for the simples of a type at
the uses, and to one that it brings in from some of the uses, and
either may have no value on a run through another use (T5-5). T5
states the fourth with the three. The instance of the meet for the
compiler's own join is owed, and its obligation is not discharged:
that every join that a meet makes is made on runs that have the four
conditions. They are hypotheses of that instance wherever the meet is
not shown to establish them, with a `CR mshinwell`: nothing in the
compiler is known to establish the first, the third and the fourth,
and X2 has an OCaml program on which the first fails and the code
compiled with the n-way join answers a wrong number.

Physical equality in the join: X3's audit finds four uses that change a
result, all in the join. The behaviour to assume of the parameter is
that of the compiler flag X3 is adding (`-flambda2-join-sharing
structural`); it does not change the contract above.

## 6. Names

Meet creates no name: there is no `Variable.create` in the 33 functions
or in `meet_env.ml`. The join does, and meet defines again, in outer
environments, variables that the join or a branch defined. The contract
speaks of names by membership only ("a variable that the environment
knows", `Typing_env.mem None`), never by number or order. On N1's
advice the model's `Meet_env.t` carries `names_to_avoid`, which the
compiler has not: the join makes variables outside it and adds them,
and meet hands it from one branch to the next so that two branches do
not make the same variable (the compiler's variables are globally
fresh). N1 asks that equivariance be stated up to a permutation of the
names created; that statement needs the renamings of types (T1),
environments (T2) and aliases (T3). Since 2026-11-12 an exact
statement is made and proved for a join that commutes with renaming
on the nose (section 16). Nothing is stated for a join that picks
new variables by looking at the names it is given.

## 7. Kinds, code ids, modes, physical equality

- Kinds: `meet_kinds`: in an environment that is not bottom, a meet of
  types of two kinds is `None` (the compiler's fatal error), never
  bottom or a type. `meet_bottom_env`: in a bottom environment the
  answer is bottom before the kinds are looked at (`Meet.meet` tests
  `is_bottom` first).
- Code ids: the model has no code age relation. Its `meet_code_id` will
  answer bottom for two different code ids; the compiler answers the
  newer when one is a later version of the other, and either input when
  it cannot tell. The contract claims nothing about inputs whose code
  ids the relation relates (T4-4), and nothing about which code a
  closure runs.
- Allocation modes and `is_unique`: transcribed, no claim.
- Physical equality (D30 as revised): in the meet it is only a
  shortcut (`meet_inverse_relations`; `reduce_head_of_kind_naked_
  immediate` compares names, which are integers; and the `Left_input` /
  `Right_input` / `Both_inputs` answers that keep sharing). X3's audit
  agrees: no use in the meet changes a result. The transcription
  follows the slow path; the contract is about the type answered
  (`extract_value`).

## 8. The step that is not sound

`reduce_inverse_relations`, case `Get_tag` (in both files): when the
numbers of a naked immediate that is `Get_tag` of a name become known,
it adds to the name the equation `blocks_with_these_tags tags`, which
has no immediate. T1's relation holds of an immediate with any number,
because unboxing records it of parameters that may be immediates and
passes 0.

- The contract is stated as it should be; T1's meaning is not changed.
- Proved: `get_tag_reduction_unsound`: a value that is not poison, a
  number that `Get_tag` relates to it, and the value has not the type
  that the step adds. On the transcribed functions:
  `compiler_meet_get_tag` (the meet answers an environment that no
  valuation that keeps the scrutinee satisfies) and
  `compiler_meet_not_sound` (so the compiler's meet has not
  `meet_sound`, whatever the join and whatever well-formedness asks,
  if the input below is well formed).
- The input on which any meet that makes this step fails `meet_sound`:
  the environment `p : Variant (get_tag = tag)`,
  `tag : (numbers unknown, Get_tag p)`, as
  `Typing_env.add_get_tag_relation` builds it; the valuation `p` = the
  tagged immediate 7, `tag` = 0, which satisfies it strictly; the meet
  of `= tag` with the type of the naked immediate 0. The value 0 has
  both types. The environment answered has
  `p : blocks_with_these_tags [0]`, which no valuation that keeps `p`
  satisfies. `example_get_tag_env_made` proves that
  `add_get_tag_relation` makes that environment.
- The change that makes it hold: build the type with
  `~immediates:Unknown` instead of the bottom type of naked immediates
  (`immediate_or_blocks_with_these_tags`). Proved:
  `get_tag_reduction_repaired`, for the tags that the step computes,
  and `meet_to_prove_get_tag` (on the same input the repaired meet
  answers an environment that the run satisfies). Nothing is lost
  where the name is already known to be a block.
- The transcription takes the type added in this case as a parameter.
  With the compiler's type it is the function T7 compares
  (`compiler_meet`); with the repaired type it is the function that is
  to be proved (`meet_to_prove`).
- Tags out of range (found by R1; T4-6). The step keeps of the numbers
  those that are tags of blocks (0 to 255) and drops the others. A
  block of the model may have any integer for its tag (that tags are
  in range is a rule of well-formed terms, D23), so a block of tag 999
  with the numbers {0, 999} would get "block with tag 0". Agreed with
  T1: `relation_holds Get_tag` relates a block to its tag only if the
  tag is in range, so the rule is used once, where a relation is made
  from a primitive, and the contract of meet has no premise for it.
  `get_tag_reduction_repaired` has no premise but the relation, now
  that T1's change is under this branch.

No miscompiled program is known (T1-1 says why T1 thinks none arises).

## 9. Precision

No claim that the meet is below its inputs. It is not, in the compiler:
an equation on a name is dropped while one is being added on that name;
a case that is impossible falls back to the case of the other tags
(`meet_knowns`, with a comment by bclement); a `Mutable_block` met with
a `Variant` loses the fields; extensions are joined. A statement of
precision for named cases can follow the functions if the reviewer
wants one.

## 10. Termination

The meet is not a recursion on its two types: a meet with an alias
meets the type that the environment has for the name. The compiler
stops because an equation on a name is ignored while one is being
added on it, and between two additions the types get smaller. Nothing
in the compiler states this.

Choice: fuel. The 33 functions are one `Fixpoint` on a `nat`; out of
fuel is `None`; the meet gives itself at the smaller fuel to the join.
The contracts are about answers that are not `None`, so soundness needs
no bound. "Some fuel is enough for every well formed input" is a
separate statement; I will attempt it after the soundness proofs, with
the measure above (names not in `adding_equations_for_names`, then the
size of the types), and say so if it does not go through. A measure in
place of fuel would need that argument before a single function could
be defined; `types_induction` does not apply, the recursion not being
on a type.

## 11. The transcription

Under the compiler's names. Each file's header says function by
function what the model has and every difference.

- `FlambdaMeetEnv.v`: the functions of `meet_env.ml` for its default
  strategy (`Dfs`: no delayed equation), with `meet_expanded_head` as
  an argument as there. The exception `Bottom_equation` is an
  `Or_bottom.t`; a fatal error is `None`.
- `FlambdaExpandHead.v`: `Expanded_type`, `expand_head_of_alias_type`,
  `expand_head0`, `expand_head`.
- `FlambdaMeet.v`: the helpers and the functions of the meet. Every
  cycle among them goes through `meet`, which is given the meet of
  expanded types as a parameter; so each function is a plain
  definition in a section, and one fixpoint on the fuel,
  `Meet.meet_expanded_head`, ties them. The parameters of what is
  exported: the join; a record of the three places where the meet can
  do other than the compiler (`meet_choices`: the type that the
  `Get_tag` step adds; whether a variable that the environment knows
  is defined again after a join, T4-1; whether the variables of a
  single case of a row-like structure are lifted, T4-9), with
  `compiler_choices` and `repaired_choices`; and the fuel.
  `Meet.meet`, `Meet.meet_shape`, and
  `Flambda2_types.add_equation`, `add_equations_on_params`,
  `add_env_extension`, `add_is_null_relation`, `add_is_int_relation`,
  `add_get_tag_relation`.
- Differences: `Dfs` only; no coercions; no arrays; the eleven
  `meet_head_of_kind_naked_*` are one; no `rec_info`; no code age;
  exceptions become results; the references of `meet_row_like` become
  a record that is handed on; names to avoid (section 6). The cases of
  blocks are gone through in the order of their tags, as in the
  compiler; function slots, value slots and the names of an extension
  in the order of the lists, which is not the compiler's (it goes by
  its maps). So the file is the compiler's algorithm on maps in
  another order, and it is not claimed that the two answer the same
  thing: the order can show in which fatal error or bottom comes
  first, in what a later meet reads, in the binding times of variables
  defined after a join, and in the order of the bindings answered; and
  giving the inputs in the compiler's order does not remove that, the
  maps and levels built on the way being lists in the order in which
  they were built. A comparison with the compiler is to be made up to
  the order of bindings and to binding times, until a statement says
  that the answer does not depend on the order.
- `FlambdaMeetGrammar.v` holds what the transcription needs of other
  streams' files, each definition to move to its owner under the same
  name: for T1, `const_of_naked_number`, `must_be_singleton`,
  `head_of_kind_value_null`, `head_of_kind_value_tagged_immediate`
  and the three functions on `set_of_closures_contents` (they are in
  T1's third request); for T2, `kind_of_simple`. What was there for
  C0 (`fold2_on`, `keys_of_both`, `merge_fold_on`), and of T1's and
  T2's what is merged (`naked_number_kind_eqb`, `block_shape_eqb`,
  `aliases_with_kinds`, `fold_on_defined_vars`), is gone: the
  transcription uses the owners' definitions.
- The functions extract. For the differential tests (T7): what is
  compared with the compiler is `Meet.meet join compiler_choices
  fuel`, with the join read as the compiler under
  `-flambda2-join-sharing structural`. It is the compiler's algorithm
  but for the `None` of T4-10, which no choice governs: on an input
  that reaches it the compiler answers and this function does not, and
  the comparison is to account for such an input (and for the order of
  the maps, above).

## 12. Discrepancies recorded

The entries of this stream:

- T4-1: after the n-way join, `meet_disjunction` and `meet_row_like`
  define again every variable of the branch levels, among them those
  that the join imported under their own names and gave equations to.
  With invariant checks it is the fatal error "Cannot rebind"; without,
  the variable gets the unknown type and a later binding time while the
  level keeps the join's equation and the aliases keep the old order.
  T5 reads `join_env.ml` the same way. Candidate; test described, not
  run. A consequence for the contract: a meet that does this does not
  keep the invariant of an environment, so `meet_keeps_well_formed`
  cannot be proved of the function as the compiler has it (soundness
  can: a variable of unknown type asks nothing). Question 4.
- T4-2: a meet through a name that holds poison (section 3).
- T4-3: a function type of a slot that the index does not have: the
  meet answers bottom for two different code ids at such a slot, and a
  closure whose set has no such slot has both types. Same root as T1-4.
- T4-4: code age (section 7).
- T4-5: a relation on a scrutinee that holds poison (section 3).
- T4-6: tags out of range in the `Get_tag` reduction (section 8).
- T4-7: `expand_head` answers the type of a constant for an alias
  whose canonical element is a constant, whatever the kind of the
  alias (`expand_head_other_kind`). So "the type expanded is of the
  kind of the type" needs a premise, `constants_of_kind`, with a
  `CR mshinwell`; it belongs with T2's invariant.
- T4-8: `is_int_for_scrutinee` says 1 for every constant scrutinee,
  `Null` included; `Relation.of_const` says 0 for `Null`.
- T4-9 (found by R1): where `meet_row_like` keeps one case only
  (`need_join` is false) it makes no extension of the case and does
  not record the variables that the environment of the case defines;
  the join is then given that one environment and keeps of its
  variables those that the equations it joins mention. A variable
  that a meet inside the case defined, and that only the type answered
  mentions, is then unknown to the environment answered, and the type
  answered does not fit it. `meet_disjunction` defines such variables
  again after its join. The transcription has the compiler's
  behaviour and the other as a choice
  (`lift_variables_of_single_case`); the contract is to be proved of
  the meet that lifts. X2 reproduced it on the compiler's own library
  of types (the variable must occur twice in the type answered).
- T4-10: `add_concrete_equation_on_canonical_name` is given a
  canonical name, meets the type to add with the type of the name,
  and gives the name the type answered, taking the name to be
  canonical still. The meet in between can have made the name an
  alias of an earlier one, if the types met lead back to the name.
  Where the type answered has one constant for only value the
  compiler then calls `Typing_env.add_alias` on a name that is not
  canonical, which `Aliases.add` forbids and does not check. The
  model answers `None` there (a `CR mshinwell` at the test,
  `replace_concrete_equation` in `FlambdaMeetEnv.v`). X5 reproduced
  it with types built by hand on the compiler's own library of types;
  whether an environment that Simplify builds can reach it is not
  known (from source it was not reached).

## 13. The questions of the first request, as answered

1. **The `Get_tag` step.** Prove the meet with the repaired type and
   keep the compiler's as a second instance for the tests. Done as
   far as the functions and the instance; the proof is next.
2. **Poison.** The handlers that unboxing makes stay outside the
   contracts. Since then D26 was revised for relations (section 3).
3. **T4-3.** Goes to the developers with T1-4, which was demonstrated
   as a miscompilation; the human has taken a fix of the n-way join of
   function types (it keeps one only if every input has one). When
   that fix is on the integration branch the n-way join keeps
   condition (b) for function types, and the entry will say so.
4. **T4-1.** Transcribed as the compiler has it, as a choice
   (`define_again` in `meet_choices`); the contract is to be proved of
   the meet that defines only what the result does not know.

## 14. What changed since the first request

From R1's review of the contract:

- `meet_expanded_head_sound` is split: the semantic part, and
  `meet_expanded_head_keeps` with the frame, the fit and the kind of
  the answer, which supposes no value. The join's contract supposes
  both.
- `scoped_env_of` lost "what satisfies the scoped environment
  satisfies `initial_tenv`"; J1 is about a valuation that satisfies
  both (section 5).
- Condition (f) is gone (section 4).
- Poison under a relation, and tags out of range (sections 3 and 8).

From T5: `type_fits` has kindedness; `typing_env_fits` has
`no_bottom_types`. The frame condition is exact agreement, as in the
first request.

From T2: its strict notion replaces my stand-in; `satisfied_no_poison`
is T2's `satisfied_no_poison_strictly`.

From the proofs of the functions of `Meet_env` and the plan for
`meet_disjunction`: `type_fits` has the kinds of the names of
equations; `meet_expanded_head_keeps` has `new_names_avoided`; the
join's contract has the premise `scoped_envs_agree` and J5
(section 5).

From R1's review of `meet_disjunction_ok`: the statements that say
that no value is lost suppose a valid static environment
(`statics_valid statics`; section 2), and so do the second halves of
`meet_call_ok` and `meet_mapping_ok` and the seven `meet_env_..._sound`
statements.

New: `add_env_extension_sound`, `add_equations_on_params_sound`,
`fits_kinded`; `compiler_meet` and `meet_to_prove`; the examples; the
statements about `expand_head` (asked for by T5b, for the type
traversal of the join): `expand_head_sound` (values are kept, under
strict satisfaction, with no other premise), `expand_head_closed`,
`expand_head_kinded`, `expand_head_kind`, `expand_head_head_of_kind`.

## 15. Next

- The contract for `meet_to_prove`, with the join's contract as a
  hypothesis, is proved: `meet_to_prove_sound`
  (`FlambdaMeetClosuresSpec.v`) is `transcription_sound
  meet_to_prove`. The last pieces were T5b's (products, heads of
  kind value, `meet_row_like` and the rows of blocks) and the rows
  of closures. What is not proved: that some fuel is enough (T1 has
  taken it), and that the compiler's own join has the contract (it
  has it only for the runs that T5's conditions describe). The
  pieces, as they came: the functions of
  `Meet_env` (`FlambdaMeetEnvSpec.v`), and the meet of the two sides
  of a variant or of a value that may be null, each met in a scope of
  its own, with the join of the two environments
  (`meet_disjunction_ok`, `FlambdaMeetScopesSpec.v`).
- `reduce_head_of_kind_naked_immediate` is sound (T5b needs it too):
  done, with no premise but those of its statement.
- The meet of two heads of naked immediates is right
  (`meet_head_of_kind_naked_immediate_ok`,
  `FlambdaMeetImmediateSpec.v`): done, for the type of the `Get_tag`
  step that is true (section 8) and a meet of expanded types that,
  beside its two contracts, adds nothing to the environment on two
  types of different kinds (`meet_expanded_head_other_kinds`; the
  transcribed meet has it at every fuel). The last is there so that
  nothing is asked of the kinds of the names of relations, on which
  a type that fits says nothing.
- The induction on fuel (`FlambdaMeetFuelSpec.v`): the step of a
  meet of expanded types that has its contracts has them, so the
  meet of expanded types has them at every fuel, and `Meet.meet` has
  `meet_contract`: done, given that the meet of two heads of kind
  value is right (`value_heads_right`: variants, row-like structures
  and closures, with the join; stated and proved apart).
  `Typing_env.add_equation` has `add_equation_sound` and
  `Typing_env.add_env_extension` has `add_env_extension_sound`;
  `add_equations_on_params` has its contract for parameters that the
  environment knows, which the contract of section 2 does not
  suppose (there is a `CR mshinwell` that says why it is needed).
- The two parameters are replaced by T1's and T2's predicates: done.
- The premise of the main theorem is one that a join has: the join
  that forgets the environments joined has the contract of a join
  (`forgetful_join_sound`), so `meet_to_prove forgetful_join fuel` has
  `meet_contract` with no premise left, whatever the fuel, and the
  functions that add to a typing environment, made of it, have theirs
  (`meet_with_forgetful_join_contract`,
  `typing_env_functions_with_forgetful_join`,
  `FlambdaMeetExamplesSpec.v`). It is not the compiler's join.
- Examples of the meet at the calls that Simplify makes
  (`FlambdaMeetMatchExamplesSpec.v`,
  `FlambdaMeetProjectionExamplesSpec.v`,
  `FlambdaMeetTagExamplesSpec.v`): a match on
  `type t = A | B | C of int`; the two fields of a pair; a float that
  is boxed and then unboxed; two value slots of a closure; a match on
  a variant with two constructors that have arguments (the arm of the
  switch on the tag is where the two meets differ); a match on a
  value that may be null. The inputs are made by hand from a reading
  of Simplify. Each
  statement gives the two types met, says that `meet_to_prove` and
  `compiler_meet` answer the same type and environment (whatever the
  join, but for the two meets of rows, which call it), and says what
  the environment answered has for the variables. The answers are
  the model's. They are inputs for the differential tests: none has
  been compared with the compiler yet.

For reviewers: whether a precision statement is wanted.

## 16. The meet under renaming (2026-11-12)

For review: `FlambdaMeetRenameSpec.v`. Asked by T1: its rewrite of
the `Switch` asks the meet, and is to commute with renaming for the
validation to use it; it took the property of the meet as a
hypothesis (`meet_equivariant`, `FlambdaSimplifySwitchSpec.v`).
`meet_function_equivariant` is that property, word for word.

What is defined. The action of a renaming on an environment of the
meet (`rename_meet_env`: the typing environment by
`rename_typing_env`, the names that an equation is being added to
and the names to avoid by the renaming of names), on what a meet
returns (`rename_return_value`: a new result is renamed, "the left
input" is not) and on its result (`rename_meet_result`). Then what
it is to commute: for a meet of two parts of a type
(`meet_of_parts_equivariant`), for a meet of expanded types
(`meet_expanded_head_equivariant`), for the join that the meet is
given (`join_scoped_envs_equivariant`: given a meet of expanded types
that commutes, of renamed environments it answers the renamed
answer; the scope is not renamed) and for a meet of types
(`meet_function_equivariant`). Each is an equation between two
options: where one side stops the other does.

What is stated, and proved.

- `meet_expanded_head_at_fuel_equivariant`: the meet of expanded
  types commutes at every fuel, for a join that commutes and choices
  whose type for the scrutinee of a tag every renaming of variables
  fixes (`meet_choices_closed`; it says nothing of symbols, code ids
  or slots, which a renaming does not touch).
- `meet_of_types_equivariant`: so `Meet.meet` does.
- `compiler_choices_closed`, `repaired_choices_closed`; so
  `meet_to_prove_equivariant` and `compiler_meet_equivariant`: the
  two meets of `FlambdaMeetSpec.v` commute, for a join that does.
- `forgetful_join_equivariant`, and
  `meet_with_forgetful_join_equivariant`: the meet that T1's rewrite
  is stated for commutes with no premise.
- `typing_env_functions_equivariant`: `add_equation`,
  `add_env_extension` and `add_equations_on_params` of
  `Flambda2_types` commute.

Why it holds, for a reader who checks the definitions against the
meet. The meet makes no variable. It compares names for equality
and reads binding times, which `rename_typing_env` keeps; it never
orders two names by their numbers. The maps of a type are keyed by
tags and slots, which a renaming leaves. The variables that it
defines again after a join are those of the level that
`Typing_env.cut` answers, in the order of that level, which
`rename_level` keeps. The one place where a renaming could show is
the join, which makes variables: hence the premise.

What is not stated.

- Nothing, in this section, for a join that does not commute on
  the nose. T5's join picks its new variables outside the names to
  avoid by a choice that reads those names, so it has not the
  premise. Section 17 is for such a join, with its source of new
  names renamed. A statement up to the choice of the new variables
  for one source of names (N1's form) is not made and not proved,
  for the join or for the meet: it needs a contract of the join
  that says which variables are new and where they may occur.
- Nothing of `Meet.meet_shape` and of the relations that
  `Flambda2_types` adds (`add_is_int_relation` and the others):
  the lemmas are there (N3's files) and no statement asks for them
  yet.

How it is proved, and by whom. N3 proved the laws of every function
of `Meet_env`, of the combinators of results, of the products, of
the entries of closures, of the naked immediates, and of the variant
and the heads of kind value given the rows
(`FlambdaEnvRenameMeet*Proofs.v`); mine are the scopes
(`add_defined_vars`, `meet_disjunction`), `meet_row_like` by the
parts that T5b's proof of its contract cuts it into, the rows of
blocks and of closures, the meet of expanded heads and the induction
on fuel (`FlambdaMeetRename*Proofs.v`). No proof needs review.

## 17. Two joins, two meets, one renaming (2026-11-12)

For review: the second part of `FlambdaMeetRenameSpec.v`. Ruled by
the orchestrator after section 16: the model's own join makes
variables, so it cannot commute with every renaming, and the
statements of section 16 say nothing of the meet that is given it.

The join of the model takes its new variables from a source of new
names, a function from the names to avoid to a variable
(`Join_env.fresh`, `FlambdaJoinEnv.v`). The correspondence that is
meant renames the source as well: the variable that the renamed
source gives for renamed names is the renamed variable. That the
join then answers the renamed answer is for the files of the join
to prove; it is not proved here. The join with the renamed source
is another function. So the relations are between two functions, for one
renaming, the second being given the renamed arguments:
`meet_of_parts_related`, `meet_expanded_head_related`,
`join_related_at` (two joins, each given its meet),
`join_scoped_envs_related` (two joins, for every two meets that the
renaming relates), `meet_function_related`.

Stated and proved:

- `meet_expanded_head_at_fuel_related`, `meet_of_types_related`,
  `typing_env_functions_related`: for two joins that a renaming
  relates, and choices as in section 16, the meets of expanded
  types, the meets of types and the functions that add to a typing
  environment are related by it, at every fuel.
- `meet_expanded_head_equivariant_iff_related`,
  `meet_function_equivariant_iff_related`: at one function, the
  relations are the properties of section 16.
- `join_related_gives_equivariant`: a join that every renaming
  relates to itself commutes with renaming. The converse is not
  stated: `join_scoped_envs_equivariant` asks the join only about a
  meet that commutes with every renaming, the relation asks it about
  every two meets that the one renaming relates.

The nine statements of section 16 keep their text. Their proofs and
those of the new statements come from the same lemmas, which are now
all for two meets and two joins (N3's four files and my four).

What is not stated.

- That a renaming relates the model's join to the same join with
  the renamed source of names. That is for the files of the join
  (N3 has started it).
- That an answer does not depend on the source of new names, up to
  the names made. Without it, the statements compare the meet on
  renamed inputs with a renamed source to the renamed answer; they
  do not compare two runs with one source.

## 18. What a meet keeps, without the values (2026-11-12)

For review: `FlambdaMeetKeepsSpec.v`; and, for what it does to the
proofs of signed statements, the device in `FlambdaMeetValidity.v`.
Asked by T5.

The need. A join of which only the half of the contract that is not
about values is known. `meet_to_prove_sound` cannot be used of the
meet that is given such a join; the half of the meet's contract that
is not about values does not need J1, and T5 asked for it alone,
for the join of the model. This series gives an interface
(`join_scoped_envs_keeps`) and a conditional theorem. The instance
for the join of the model is neither proved nor used here: it is in
the separate work on the join (T5's and T5b's files).
What that work has or has not of J1, and whether the conditions it
names become premises of J1 or the join changes, is not decided
here.

What is defined: `join_scoped_envs_keeps`, the contract of the join
without the premise `meet_expanded_head_sound meet_head` and without
J1; the text is T5's, which T5 deletes from its file for this one.

What is stated and proved:

- `meet_expanded_head_fuel_keeps`: with a join that has
  `join_scoped_envs_keeps`, and choices that have the three
  conditions of the main theorem, `Meet.meet_expanded_head join
  choices fuel` has `meet_expanded_head_keeps` at every fuel.
- `meet_expanded_head_repaired_keeps`: so at `repaired_choices`.
- `forgetful_join_keeps`, `meet_with_forgetful_join_keeps`: the join
  that forgets has the definition, so the premise can be had.
- `meet_of_types_keeps`: with the same premises, the meet of types
  `Meet.meet join choices fuel` has `meet_keeps_well_formed`, the
  part of `meet_contract` that says its answers may be met again.
- `meet_of_types_kinds`: the two other parts of `meet_contract` that
  are not about values, `meet_kinds` and `meet_bottom_env`, hold of
  `Meet.meet join choices fuel` whatever the join, the choices and
  the fuel.
- `typing_env_functions_keep`: the half that is not about values of
  the three functions that add to a typing environment
  (`Flambda2_types.add_equation`, `add_env_extension`,
  `add_equations_on_params`), for a meet of expanded types that has
  `meet_expanded_head_keeps`. The signed contracts
  (`add_equation_sound` and the two others) say that the environment
  answered is satisfied and not that it fits, which they suppose
  of the environment given; so they alone do not give what is
  needed to use them again of an environment answered (this is
  about what can be proved from the contracts, not about whether a
  second call answers). Here: the
  environment answered fits, and what fitted the first, was of a
  kind in it or was known to it is so in the second
  (`typing_env_kept`, the second definition of the file). The
  premises are those of the functions of `Meet_env` that the three
  are made of: what is added fits, and the name of an equation is of
  the kind of its type (`simple_of_kind`, `env_extension_kinded`),
  which the contracts about values have from the value that the
  name holds. The proof is the three signed statements of
  `FlambdaMeetEnvSpec.v` about those functions, read of the typing
  environments.

What is not stated: nothing about values; nothing that is kept for
`compiler_choices` (of the meet with those choices only
`meet_of_types_kinds`, which is for any choices, says anything);
nothing of the functions that add a relation
(`Flambda2_types.add_relation` and the three made of it); and
neither of
`join_scoped_envs_sound` and `join_scoped_envs_keeps` is shown to
imply the other (the first gives its conclusions only for a meet
that loses no value).

How it is proved, which a reviewer of the signed statements may
want to check. Above the layer of `Meet_env`, the statements of the
contract bundle the two halves (`part_meet_ok`), so the keeping
half was not a corollary. But every clause about values supposes
`statics_valid statics`, and no proof of the nine files looks into
it: they hand it on, and in one place to a fact about the closures
of a set. `FlambdaMeetValidity.v` has a class of predicates on
static environments that imply validity, and, made by a tactic from
the signed definitions, copies of `meet_expanded_head_sound`,
`join_scoped_envs_sound`, `meet_call_ok`, `part_meet_ok` and of four
definitions made of them, with the predicate where they have
`statics_valid`; at `statics_valid` each copy is the signed
definition by `reflexivity`. In the nine proof files the lemmas are
now in a section that has such a predicate, in which the signed
names stand for the copies; no proof script changed but one line.
Each `Theorem X : X_statement` is, after the section, the lemma of
the section at `statics_valid`. So:

- the signed statements are not changed, and each is still proved
  by a theorem of its name in its proofs file; `make assumptions`
  is as it was;
- at the predicate that no static environment has, J1 and every
  other clause about values is vacuous, the join's contract is
  `join_scoped_envs_keeps`, and the chain gives
  `meet_expanded_head_keeps`.

Three of the nine files are T5b's (`FlambdaMeetProductsProofs.v`,
`FlambdaMeetHeadsProofs.v`, `FlambdaMeetRowLikeProofs.v`); T5b
agreed to the change being made in this series.

## 19. The meet of expanded types of the meet that is proved (2026-11-12)

For review: one statement added to `FlambdaMeetClosuresSpec.v`,
`meet_to_prove_expanded_heads_statement`: for every join that has
`join_scoped_envs_sound` and every fuel, `Meet.meet_expanded_head
join repaired_choices fuel` has `meet_expanded_head_keeps` and
`meet_expanded_head_sound`.

The need. These two are the premises of `meet_of_fuel_contract`
and `typing_env_functions_sound` (`FlambdaMeetFuelSpec.v`) and of
`typing_env_functions_keep` (section 18). Until now a file that
wanted them for a join that has its contract took them from a lemma
of `FlambdaMeetClosuresProofs.v` (`value_heads_of_join`) and
`meet_expanded_head_fuel_ok`; two files of the join's side do. The
statement says nothing new: its proof is the first three lines of
the proof of `typing_env_functions_of_meet_to_prove`.

## 20. Parameters that the environment does not know (2026-11-12)

For review: `FlambdaMeetParamsSpec.v`.

The question was whether `add_equations_on_params_sound`, as agreed
without a premise on the parameters, is true of the model's
function (discrepancy T4-11; "Decided overnight", 16, proved it for
parameters that the environment knows).

It is false, for a join that has `join_scoped_envs_sound`:

- `naming_join q`: the join that forgets, and that makes the
  variable `q` with the type of the naked immediates 5 and 6 if no
  environment that it is given knows `q` and `q` is not to be
  avoided. `naming_join_sound`: it has the contract of the join.
- The example: an environment that knows the first of two
  parameters (a value that is an immediate or a block) and not the
  second; the first is given the type "an immediate or a block of
  tag 0", whose meet with what the environment has calls the join
  once; the second holds 7 and is given the type of 7 and 8.
  `example_params_premises`: every premise of the contract holds.
  `example_params_answer`: the function answers the bottom
  environment. `add_equations_on_params_not_sound`: so it has not
  the contract.

It is a fault of the model's supply of names and not of the
compiler, whose variables differ from every variable that exists.

The premise under which the contract holds:

- `meet_env_add_equations_on_params_avoided`: for
  `Meet_env.add_equations_on_params`, every parameter that the
  environment does not know is among the names that the meet is to
  avoid. Then such a parameter stays unknown (a variable that a meet
  defines is not among the names to avoid) and its equation adds
  nothing: `add_equation_on_unknown`, where the function answers
  the environment is as it was. It need not answer: the compiler
  stops with a fatal error on a name that is not bound.
- For the function as transcribed, which starts with no name to
  avoid, this is that every parameter is known: the statement that
  was proved.
- `add_equations_on_params_avoiding`: the function that starts with
  the parameters as names to avoid. It is not the transcription.
  `add_equations_on_params_avoiding_sound`: it has the contract as
  agreed, for a join that has its own.
  `example_params_avoiding_answer`: on the example it does not
  answer. That is one input; it is not shown that the function
  never answers where a parameter is not known. What is shown in
  general is `add_equation_on_unknown`: of such a parameter the
  environment answered says nothing.

A premise on the join alone is not stated; the header says why, and
what the join of the model does (it makes the variable one greater
than every name it is to avoid and every variable its environments
know, so it can make a parameter that the environment does not
know; with the parameters to avoid it cannot).

What to check: that `naming_join` is a fair instance of the
contract (it is `forgetful_join` plus one definition and one
equation, and its proof uses nothing but the laws of the typing
environment); that the example's input is built by the model's own
functions; that `add_equations_on_params_avoiding` differs from
`Flambda2_types.add_equations_on_params` only in the names to avoid
of the environment it starts with.

Which of the two the model should have (the contract with its
premise, or the function that avoids the parameters) is for the
human: status, "Decided overnight", 21.

## 21. The compiler's three choices (2026-11-12)

For review: `FlambdaMeetChoicesSpec.v`.

The main theorem is for `repaired_choices`. For each of the three
places where `compiler_choices` differs, with that one choice as the
compiler has it and the two others repaired (`only_get_tag_of_blocks`,
`only_defining_again`, `only_not_lifting`):

1. The type of the scrutinee of `Get_tag` (T4-6).
   `get_tag_reduction_of_block`: the compiler's type is true of a
   value that the relation holds of exactly where the value is a
   block. `example_get_tag_inputs`: the input of the example that
   was in the tree is one that the contract speaks of, which the
   statement there supposes. `only_get_tag_not_sound`: the meet with
   this choice alone has not `meet_sound`, whatever the join;
   `compiler_meet_not_sound_outright`: nor has the compiler's meet,
   now with no premise.
2. Defining again (T4-1). `define_in_types_choice`: the two choices
   do the same on a variable that the environment does not know; on
   one that it knows the repaired choice does nothing and the
   compiler's defines it. `example_rebind`,
   `defining_again_not_keeps`: with the join of the model as the
   compiler has it, on the input of the demonstration made with the
   compiler's library of types, the environment answered is not well
   formed (a variable at two binding times); what fails is
   `meet_keeps_well_formed`. `example_rebind_inputs`: the input is
   one that the contract speaks of. `example_rebind_repaired`: with
   the repaired choice the environment answered is well formed.
3. Lifting the variables of a case that is alone (T4-9).
   `union_extra_variables_same`: recording the variables of the
   case's environment changes nothing exactly where each is recorded
   already. `example_unbound`, `not_lifting_not_keeps`: on the input
   of the differential test's scenario, the type answered mentions
   a variable that the environment answered does not know.
   `example_unbound_inputs`, `example_unbound_repaired` as above.

The comments of the statements say which faults the compiler shows
only with types built by hand (the second and the third).

Not stated, and the header says why: one theorem "on inputs with P
the compiler's meet has the contract". For the second and the third
place the condition is about what the join makes or brings in at
each call, not about the types met. Nothing either for the state at
which the model stops and the compiler does not (T4-10): what would
exclude it is that the name is still canonical after the meet, and
to state that the function then answers needs that adding an alias
of a canonical name to a constant answers, which no statement about
aliases says yet.

What to check: that each `only_` record differs from
`repaired_choices` in one field; that the third statement of each
group follows from the second and the inputs statement and says
"not `meet_keeps_well_formed`" of the meet with the model's join,
of which nothing is supposed; that the example of the second group
is the input of the demonstration (two variables that are tagged
immediates, tags 0 and 1 against tags 0 and 2).
