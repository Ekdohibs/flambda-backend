# What units made from OCaml sources need that the model has not got

From V2 (the corpus), for whoever adds constructs to the model and
rewrites to the validator. The units are made by
`middle_end/flambda2/tests/semantics/ocaml/from_ocaml.sh`
(`-drawfexpr`); each fragment below is what the front end prints.
Ranked by how much OCaml each one blocks: the first ones are in almost
every program. The state of each is in the last column (2026-11-12,
after V4's validation of loops and the removal from the term after of
the continuations that are applied once);
`tests/semantics/ocaml/README.md` has the units and what stops
each.

Three states of a unit, as the test tool reports them: **not checked**
(the conversion refuses a construct), **the same, not validated** (the
two runs agree, the validator cannot match the terms), **undefined
before** (the model has no step for the source, so anything passes: the
worst of the three, as it looks like a pass).

## 1. Not checked: constructs and primitives

| Rank | Construct | OCaml that makes it | Fragment | State |
|------|-----------|---------------------|----------|-------|
| 1 | `Phys_equal` on immediates | `=` and `<>` on `int`, a match on an integer constant, `if n = 0` | `let prim = %phys_eq (n, 0) in let eq = %tag_imm (prim) in let untagged = %untag_imm (eq) in switch untagged ...`; `%phys_ne` for `<>` | P1, done |
| 2 | Regions and local allocation | an argument or a closure that does not escape, a local function | `let region = %begin_region () in let ghost_region = %begin_ghost_region () in ... let b = %block.[0].toplevel.local[region] (x) in ... let unit = %end_region (region) in let unit = %end_ghost_region (ghost_region) in ...`; a closure: `closure g_8 @g &my_alloc_region &region with { x = x }`; a call: `apply direct(f &toplevel &region &ghost_region) ...` | dropped by the conversion (D37) |
| 3 | Trap actions and `raise` | integer division and `mod` (even by a constant), `assert`, array access, `raise`, `try` | `(let prim = %int_comp.ne (4, 0) in switch prim \| 0 -> k1 \| 1 -> k2) where k1 = cont error pop(regular error) ($`*predef*`.caml_exn_Division_by_zero)` | C5, done: the model has trap actions; what Simplify does to them is in section 3 |
| 4 | `Boolean_not` | `not b` | `let Pnot = %boolean_not (b) in ...` | P1, done |
| 5 | `Num_conv` between integer kinds | `for` loops (the counter is a naked `int64`), `Int64.of_int`, `Int64.to_int` | `let prim = %untag_imm (n) in let c = %num_conv.[imm].[int64] (prim) in ...`; `let prim = %num_conv.[int64].[imm] (counter) in let i = %tag_imm (prim) in ...` | P1, done |
| 6 | `Int_shift` | `lsl`, `lsr`, `asr` | `let prim = %untag_imm (2) in let s = %int_shift.lsl (x, prim) in ...` | P1, done |
| 7 | Float arithmetic and comparison | `+.`, `<` on floats | `let a = %unbox_num.float (x) in let b = %unbox_num.float (y) in let c = %bfloat_arith.add (a, b) in let r = %box_num.float.toplevel (c) in ...` | P1's request for floats |
| 8 | Mutable blocks | a reference or a mutable record field that escapes | `%block_load` with `Mutable`, `%block_set` | C4: the heap, first step done; the machine does not run its primitives yet (`ref_counter.ml`) |
| 8a | A load marked mutable of an immutable block | `%field0` and `%field1` written by hand on a tuple (the standard library's `fst` and `snd` are `%field0_immut` and `%field1_immut`, an immutable load, which the model has) | ``let Pfield = %block_load.mut.[`0`] (p) in ...`` (unknown tag and size) | P1, done |
| 8b | `String_length`, a load of a byte of a string, the unary `not` of integers | `String.length`, `String.unsafe_get`, `lnot` (Simplify makes `x lxor (-1)` the unary primitive) | ``%string_length (s)``, ``%string_load.[`8`] (s, i)`` | P1, done, with the wider loads |
| 9 | Arrays | `a.(i)` | `Array_load`, with a bounds check that raises | P1, on C4's heap; needs the heap in the machine |
| 10 | C calls | polymorphic comparison, `exception E` (`caml_fresh_oo_id`), anything of the runtime | `apply ccall ($`*extern*`.caml_equal : val * val -> val) (x, y) -> k * error`; this does not parse back (below) | P1's request for calls of C functions |
| 11 | The constant `poison` | a variant that Simplify unboxes where two branches meet (the term after only) | `#poison[variant_unboxing_field]` | done: converted to the poison constant of the model |

Not blocking, because the conversion drops them (a hack, with CRs):
`let next_depth = rec_info (succ my_depth)` at the head of every
function, `my_depth`, and the coercion `~ depth my_depth -> next_depth`
on the callee of a recursive call.

On 2: stack allocation cannot be turned off by a flag (it is fixed when
the compiler is configured). A type annotation on the function keeps
its argument on the heap, which is how the first examples avoided it.
In the terms seen, a region always comes with a local allocation in
it. The conversion now drops regions and converts a local allocation
as one on the heap; the README of the examples says what that
assumes.

On 8 and 9, since the column was written (2026-11-15): the machine runs
the primitives of the heap, and the conversion of the test tool
makes them in the cases that the README of the examples lists ("What
the verdicts mean", the item "not checked"). `ref_counter.ml` and
`array_literal.ml` are validated.

## 2. Undefined before: the model has no step for a defined program

| Construct | OCaml that makes it | What the model does | What it should do |
|-----------|---------------------|---------------------|-------------------|
| A tupled function | `let f (x, y) = ...` called through its closure | the code has two parameters and `tupled`; the call passes one block; the conversion dropped `tupled` and the machine found an arity mismatch. Now a unit with tupled code and a call that is not direct is refused | a mark on code, and the rule for a call of unknown arity |
| Partial application through an unknown function | `let app f x = f x` then `app add 1` | done (C3): a call with fewer arguments than parameters is a partial application, a value of the model | |
| Over-application through an unknown function | `let app2 f x y = f x y` then `app2 adder 1 2` | done (C3): a call with more arguments than parameters calls with the first, then calls the result with the rest | |

(Where the function is known when the term is made, the front end
splits an over-application into two calls and makes a stub for a
partial one, and these do run.)

## 3. The same, not validated: rewrites the validator has not got

| Rank | What Simplify does | OCaml that shows it | State |
|------|--------------------|---------------------|-------|
| 1 | Puts the handler of a continuation applied once where it is applied, when that is in the handler of another continuation | every unit whose top has an `if` or a `match` on a value that is not known (`if_unknown.ml`) | done: V4's `sink_let_conts` |
| 2 | Lifts a closed closure, block or boxed number at toplevel to a symbol | any function that stays (recursive, not inlined, or exported), any block that stays | done: V3's lifting |
| 3 | Leaves the calls of a recursive function | `factorial.ml`, `list_sum.ml` | done: the validator is told the recursive code and does not inline its calls (`kept`) |
| 4 | Rewrites inside the handler of a recursive continuation; removes a parameter of a continuation that is not used; replaces a continuation that is one `Apply_cont` by it in a `Switch` arm; follows a constant bound outside a loop into it | `while` and `for` loops | done: V4's series (the rewrites of a round inside loops; a parameter that every application gives the same thing; the unused parameters of loops; a parameter that is only passed round the loop, `FlambdaDeadParamsSpec.v`) |
| 5 | Makes a self tail call a jump (loopify) | tail-recursive functions | done: V4's `loopify_unit`; `validate_unit` takes the code to loopify, with a name for each loop, as one more guess, and the tool gives the code that the front end marks (`tail_loop.ml`) |
| 6 | Identities of arithmetic (`x + 0`, `x * 1`, `x land 0`, ...; not `x - x`) | `add_zero.ml` | done: P1's `FlambdaIdentitiesSpec.v`, wired |
| 7 | Loads a field, the tag or `Is_int` of a known block | every unit (the module block), every match on a constructor just applied | done: `FlambdaBlockLoadSpec.v` |
| 8 | Deletes a continuation that is not used | every match resolved | done: `FlambdaUnusedContsSpec.v` |
| 9 | Unboxes a number it has seen boxed, untags one it has seen tagged | arithmetic on `int64`, every conditional | done: `FlambdaUnboxSpec.v` |
| 10 | Gives fresh names to what it inlines | a function inlined twice (`twice_inlined.ml`) | done: `FlambdaRenameApartSpec.v`, every binder of the term before given a name of its own once calls are inlined |
| 11 | Shares a primitive applied twice to the same arguments | a boxed number used twice (`nativeint_arith.ml`) | done for pure primitives and for the loads marked immutable (`list_last.ml`): `FlambdaCseSpec.v`; not for immutable allocations, which Simplify shares too |
| 12 | Binds the code of functions that call each other in another order than the front end | `mutual_recursion.ml` | done: `FlambdaStaticsOrderSpec.v` |
| 13 | Computes a primitive applied to a symbol that the unit binds to a constant | `String.length` of a string constant (`string_length_const.ml`) | done: V3 (known symbols) |
| 14 | Lifts a set of closures one of whose value slots is a variable | a local `let rec ... and ...` closed over an argument, in a function inlined at toplevel (`closed_mutual_recursion.ml`) | done: V3 (the function slots of a static set are known whatever its value slots hold; liftings in turn) |
| 15 | Makes an application that only goes on to another go there, trap action included; makes the handler of a `try` that nothing raises to any more an ordinary continuation | division, checked accesses, `try` (`div_unknown.ml`, `try_no_raise.ml`) | done: V4's `forward_applications`, and C5's `demote_handlers` at the end of the normalisation of the term before (`FlambdaValidateSpec.demote_and_tidy`) |
| 16 | Unboxes a value where two branches meet: passes the fields of a pair, or of a variant with `poison` for the fields a branch has not got | `pair_join.ml`, `option_join.ml` | done where each branch builds the value: T6's `FlambdaUnboxVariantsSpec.unbox_variant_params`, in the first phase of the term before. Where a branch gives a parameter of the function Simplify goes by the subkind of the parameter, which the model has not got (`pair_join.ml`; discrepancy T6-7) |
| 17 | Inlines a call that becomes direct in the body of code, of code bound in the same group | a local function given to a function that is inlined, in a function that stays (`closure_argument_exported.ml`) | done: `FlambdaInlineSiblingsSpec.v` |
| 18 | Nothing: the compiler binds every closure of a set in one `Let`, and the model's `Let` of the ones that are not used is a projection that the term after keeps | `let rec g ... and h ... in g` (`returned_closure_of_pair.ml`) | done: `FlambdaUnusedProjectionsSpec.v`, for the term after |
| 19 | Leaves a continuation that is applied once where the validator, which inlines more calls, has one in the term after only | units with inputs that are not known (50 of 600 generated units) | done: `FlambdaValidateSpec.inline_once_applied_after`, by V4's `inline_tail_conts_kinded`, an equivalence |
| 20 | Makes a `Switch` whose two arms give one continuation 0 and 1 the tagging of what is tested | a boolean computed by a test and kept (3 of 600 generated units have it; two of them validate with the rewrite, T1's measure) | T1's `FlambdaMergeSwitchesSpec.merge_switches`, merged; to be wired after this |
| 21 | Makes of a partial application a wrapper that calls the function through a value slot of its own closure, and leaves the call indirect | `partial_exported.ml` | V3 |
| 22 | Keeps a boolean untagged in a parameter of a loop | `while_flag.ml` | T6 |
| 23 | Leaves, in the code of a function that it makes a loop of, a `Let` of a parameter of the code that nothing uses | a loop over a string in a function (`string_sum.ml`; 3 of 600 generated units) | done: V4's `FlambdaCodeLetsSpec.v`, for the term after |
| 24 | Resolves a `Switch` on a variable that a `Switch` around it has tested | a condition used twice (generated units) | T2: `prune_by_types`, to be wired |
| 25 | Makes a block a symbol when the handler that made it only made it and went on, and drops the handler | a constructor applied in one arm of a match, in a value that the unit keeps (generated units) | V3: the guess of what was lifted; and a round of clean-up after the unused static bindings are removed |

## 4. The printer and the parser of fexpr

- A C call is printed `apply ccall [noalloc] (...)` and the parser
  wants `ccall [noalloc] &<region>`: a unit with a C call that
  `-drawfexpr` prints is a syntax error when read back. The same for a
  method call.
- `Make_block` is printed without the kinds of its fields, and read
  back with `any_value` for each. So the term the tool gives to
  Simplify is not the term the compiler gives it: it says less of the
  fields. The conversion's refusal of a field kind other than
  `any_value` never fires on a term that was parsed.
- A direct call is printed without the arity of its parameters.
