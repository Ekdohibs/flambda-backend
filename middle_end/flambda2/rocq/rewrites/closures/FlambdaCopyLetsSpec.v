(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* A copy of code for the sets of closures that a [Let] makes: where
   Simplify has given such a set new code, the term before is given a
   copy of the old code under the new code ids, and the [Let] makes
   its closures of the copy. A normalisation that refines the unit run
   on its own, where no closure reaches an exit.

   This file is to be reviewed: [copy_named], [copy_lets_expr],
   [copy_lets_in_code] and [copy_let_code], against what Simplify does
   (below); the two tests that [copy_let_code] makes of the unit and
   the one that it makes of the copy ([copies_apart]); and the
   statements, the last two of which are the rewrite computed on
   a small unit. [with_let_copies] is extracted and run by the
   validation of middle_end/flambda2/tests/semantics/, in the term
   that FlambdaValidateCopyingSpec.before_copied makes of the term
   before; no function of FlambdaValidateSpec.v calls it.

   What Simplify does. It gives new code to every set of closures that
   it simplifies, wherever the set stands
   (FlambdaCopyCodeSpec.v has the places in the compiler). A set that
   Simplify meets twice has two newer versions of its code: once in
   the body of a function that stays, and once in a copy of that body
   that it inlines. Where Simplify lifts the closure of the copy to a
   symbol, FlambdaCopyCodeSpec.copy_static_code gives that static set
   the copy. Where it does not (a value slot of the set holds a
   variable, for one), the set stays a [Let], and the term after has,
   at the place of one [Let] of the term before, a [Let] that makes
   its closures of code under a new id.

   What is here. [copy_let_code (ids, symbols, (lets, in_code)) e] is
   FlambdaCopyCodeSpec.copy_static_code of [ids] and [symbols], which
   binds the copies and makes indirect every direct call that names
   copied code and whose callee is no symbol; and then
   - each [Let] of the unit outside code that stands at a path of
     [lets] and makes a set of closures, or a closure of a set, makes
     it with the copy of each of its code ids ([copy_named]);
   - the same of each [Let], in the body of a piece of code that
     [in_code] names, that stands at one of the paths given with that
     code. The paths are paths in the body, from its root, and the
     code is named by the id that the unit binds it to: the body of a
     copy that this same rewrite makes is entered only if [in_code]
     names the new id.
   A path is that of FlambdaLiftSpec.lift_expr: where the [Let]
   stands in the term. [with_let_copies] makes each of the copies
   given, in turn; the paths of a copy are paths in the term that the
   copies before it give, in which every [Let] stands where it stood.

   Why it is not a rewrite of every unit. A closure that a [Let]
   makes is a value that holds its code ids
   (FlambdaValueSpec.Value_closure_of_set). So the two runs differ in
   the closures of the [Let]s copied, and in every value that holds
   one: FlambdaCopyCodeSpec.copy_static_code leaves those [Let]s
   alone for that reason. The copy has the text of the code that it
   copies, and a closure of the copy is called, projected from and
   passed about as the closure of the old code is. What tells the two
   apart is an exit that gives the closure, a cell of the heap that
   holds it at the exit (FlambdaSemanticsSpec.refines compares the
   heaps of two exits by equality; decision D40), and a primitive
   that reads the code of a closure. So [copy_let_code] makes two
   tests of the unit, those that a lifting to a symbol makes for the
   same reason (FlambdaLiftSpec.lift_unit), and is the identity where
   one fails:
   - the unit exits with symbols and constants only
     (FlambdaLiftSpec.exits_plain_knowing);
   - the unit has no primitive of the heap
     (FlambdaHeapFreeSpec.expr_free_of_heap: the guard that the
     liftings have for now, with the same limit and the same CR, at
     FlambdaLiftSpec.lift_unit).
   No primitive reads the code of a closure as the model stands:
   FlambdaPrimitiveSpec.eval_phys_equal has no value on two allocated
   values (the CR there). If it were given one on two closures that
   depended on their code ids, the statements here would be false: a
   closure of the copy and a closure of the old code that are equal
   in the unit would be told apart in the result.

   It makes a third test, of the copy and not of the unit: no code id
   that is given a copy is itself among the new ones
   ([copies_apart]). It is a limit of the proof (the CR there), and
   the identity where it fails.

   It is right whatever [ids], [symbols], [lets] and [in_code] are,
   where the three tests pass and the check of
   FlambdaCopyCodeSpec.copy_static_code passes (the old code ids are
   distinct, the new ones are distinct, and the unit binds none of
   the new): which [Let]s and which code is a guess of the caller's,
   read off the term after, and nothing is believed of it. A [Let] at
   a path given that makes no set of closures is left as it is; a
   code id that [ids] gives no copy is left as it is.

   What is stated. The result refines the unit run on its own, and
   run from its imports ([copy_let_code_program_refines_statement],
   [copy_let_code_refines_imports_statement], and the two of
   [with_let_copies]). Nothing is asked of the imports: they bind no
   code, and an import is no closure.

   What is not stated, and is not so. The result does not refine the
   unit under all environments (FlambdaSemanticsSpec.expr_refines): a
   static environment may bind a new code id already, and may bind
   code that the two tests have not seen, which gives to an exit a
   closure that it is passed, or stores it. Nor is it stated of the
   unit followed by what may follow a unit (FlambdaObserversSpec.v),
   as it is for the copy of static sets: what follows may read a
   closure that the unit made from a symbol that the unit gave it,
   and exit with it. Nothing is stated of a unit that has a primitive
   of the heap, of which the rewrite is the identity.

   The rewrite commutes with renaming, which renames no code id and
   no symbol, and names a [Let] by where it is
   ([copy_let_code_equivariant_statement]); that it keeps scope is in
   FlambdaValidateCopyingSpec.v.

   The proofs are in FlambdaCopyLetsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaUnusedStaticsSpec
  FlambdaImportsSpec FlambdaLiftSpec FlambdaHeapFreeSpec
  FlambdaCopyCodeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The [Let]s given make their closures of the copies                       *)
(* ======================================================================== *)

(* A set of closures, or a closure of a set, with the copy of each of
   its code ids (FlambdaCopyCodeSpec.copy_of: the id itself where
   [ids] gives it no copy). Any other defining expression is as it
   was. *)
Definition copy_named (ids : list (code_id * code_id)) (d : named) : named :=
  match d with
  | Set_of_closures code value_slots =>
    Set_of_closures (copy_of ids code) value_slots
  | Closure_of_set slot function_decls value_slots =>
    Closure_of_set slot (copy_decls ids function_decls) value_slots
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    d
  end.

(* The term with the [Let]s at the paths given so changed. [at_path]
   is the path of the term; the paths are those of
   FlambdaLiftSpec.lift_expr. No static binding is entered, so no
   body of code is. *)
Fixpoint copy_lets_expr (ids : list (code_id * code_id)) (lets : list path)
    (at_path : path) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (if among at_path lets then copy_named ids d else d)
      (copy_lets_expr ids lets (xO at_path) body)
  | Let_static bindings body =>
    Let_static bindings (copy_lets_expr ids lets (xO at_path) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (copy_lets_handler ids lets (xI at_path) h)
      (copy_lets_expr ids lets (xO at_path) body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (rec_handlers_map (copy_lets_handler ids lets) (xI at_path) handlers)
      (copy_lets_expr ids lets (xO at_path) body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with copy_lets_handler (ids : list (code_id * code_id)) (lets : list path)
    (at_path : path) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (copy_lets_expr ids lets at_path handler)
  end.

(* The same in the body of a piece of code that [in_code] names, with
   the paths that it gives for that code, which are paths in the body
   from [root]. Code that it does not name is as it was. *)
Definition copy_lets_binding (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id (Code params my_closure ret exn metadata body) =>
    match lookup id in_code with
    | Some lets =>
      Bind_code id
        (Code params my_closure ret exn metadata
           (copy_lets_expr ids lets root body))
    | None => binding
    end
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

(* That, of every static binding of the unit outside code, in
   handlers too. The body of code is not looked into for bindings of
   code: the code of a well-formed unit binds nothing static. *)
Fixpoint copy_lets_in_code (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (copy_lets_in_code ids in_code body)
  | Let_static bindings body =>
    Let_static (map (copy_lets_binding ids in_code) bindings)
      (copy_lets_in_code ids in_code body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (copy_lets_in_code_handler ids in_code h)
      (copy_lets_in_code ids in_code body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, copy_lets_in_code_handler ids in_code h))
         handlers)
      (copy_lets_in_code ids in_code body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with copy_lets_in_code_handler (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (copy_lets_in_code ids in_code handler)
  end.

(* ======================================================================== *)
(* One copy for [Let]s, and the copies in turn                              *)
(* ======================================================================== *)

(* What one copy is given: the code ids, each paired with a new one;
   the symbols of the static sets of closures that get the new ones,
   as for FlambdaCopyCodeSpec.copy_static_code; the paths of the
   [Let]s outside code that get them; and, for the bodies of code,
   code ids, each with the paths of the [Let]s of its body that get
   them. *)
Definition let_copy : Type :=
  (list (code_id * code_id) * list symbol
   * (list path * list (code_id * list path)))%type.

(* No code id that is given a copy is among the new ones: no pair
   asks for the copy of a copy. *)
(* CR mshinwell: this is a limit of the proof, and is not known to be
   a condition of the rewrite. Where [ids] pairs a code id with a
   second and the second with a third, a [Let] that makes a closure
   of the second is given the third, which nothing binds:
   FlambdaCopyCodeSpec.copy_static_code binds the second, as the copy
   of the first, and no copy of it, the unit binding none of the new
   ids. In the unit that closure cannot be called, the second not
   being bound there; in the unit with its static copies, which the
   proof goes by, it can. A pair whose old code id is among the new
   ones copies no code, for the same reason, so a caller loses
   nothing by leaving such a pair out. The test is of the whole
   list: where one pair fails it the rewrite is the identity, and
   the copies that the other pairs ask for are not made either. *)
Definition copies_apart (ids : list (code_id * code_id)) : bool :=
  forallb (fun id => negb (among id (map snd ids))) (map fst ids).

(* The unit with one copy, if it exits with symbols and constants
   only, has no primitive of the heap, the copy passes
   [copies_apart], and the check of
   FlambdaCopyCodeSpec.copy_static_code passes; the unit as it is
   otherwise. The first two tests are of the unit given. *)
Definition copy_let_code (copy : let_copy) (e : expr) : expr :=
  let '(ids, symbols, (lets, in_code)) := copy in
  if exits_plain_knowing [] [] e && expr_free_of_heap e && copies_apart ids
  then
    match copy_static_code ids symbols e with
    | Some e' =>
      copy_lets_in_code ids in_code (copy_lets_expr ids lets root e')
    | None => e
    end
  else e.

(* Each of the copies given, in turn. *)
Fixpoint with_let_copies (copies : list let_copy) (e : expr) : expr :=
  match copies with
  | [] => e
  | copy :: copies => with_let_copies copies (copy_let_code copy e)
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What [copy_let_code] gives refines the unit, run on its own from
   empty environments. Nothing is asked: where a test fails it is the
   unit. *)
Definition copy_let_code_program_refines_statement : Prop :=
  ∀ (copy : let_copy) (e : expr), program_refines e (copy_let_code copy e).

(* The same of a unit that imports symbols, run from the static
   environment of its imports. Nothing more is asked: the imports bind
   no code, and an import is no closure. *)
Definition copy_let_code_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (copy : let_copy) (e : expr),
    unit_refines imports e (copy_let_code copy e).

(* With no copy asked for, the term is as it was. *)
Definition with_let_copies_none_statement : Prop :=
  ∀ e : expr, with_let_copies [] e = e.

(* The two, of each of the copies in turn. *)
Definition with_let_copies_program_refines_statement : Prop :=
  ∀ (copies : list let_copy) (e : expr),
    program_refines e (with_let_copies copies e).

Definition with_let_copies_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (copies : list let_copy)
    (e : expr),
    unit_refines imports e (with_let_copies copies e).

(* The rewrite commutes with renaming, which renames no code id and no
   symbol and keeps every [Let] where it is; the two tests of the
   unit answer the same of a term and of the term renamed. *)
Definition copy_let_code_equivariant_statement : Prop :=
  ∀ (r : renaming) (copy : let_copy) (e : expr),
    rename_expr r (copy_let_code copy e) = copy_let_code copy (rename_expr r e).

Definition with_let_copies_equivariant_statement : Prop :=
  ∀ (r : renaming) (copies : list let_copy) (e : expr),
    rename_expr r (with_let_copies copies e)
    = with_let_copies copies (rename_expr r e).

(* ======================================================================== *)
(* One [Let] copied, and one unit left as it is                             *)
(* ======================================================================== *)

(* What the rewrite makes of a small unit, computed. One piece of
   code is the code of two closures, each made by a [Let]; the unit
   calls the first by a direct call and exits with a constant. The
   copy is asked for the first [Let], by its path: the static binding
   of the code is at the root, and the [Let] is its body. *)
Section Examples.

Let x : variable := 1%positive.
Let my_closure : variable := 2%positive.
Let first_made : variable := 3%positive.
Let second_made : variable := 4%positive.
Let result : variable := 5%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 3%positive.
Let failed : continuation := 4%positive.
Let returned : continuation := 5%positive.
Let old : code_id := 1%positive.
Let new : code_id := 2%positive.

Definition let_copied_arity : arity := [Singleton Value].

(* The code: it returns its argument. *)
Definition let_copied_code : code :=
  Code [x] my_closure ret exn
    (make_code_metadata let_copied_arity let_copied_arity)
    (Apply_cont (Apply_cont_expr ret [Var x] None)).

(* The unit, with the bindings of code given, the code given for each
   of the two closures, the kind given for the call of the first, and
   what it exits with given. *)
Definition let_copied_unit (code_bindings : list static_binding)
    (first second : code_id) (kind : call_kind) (exits_with : simple)
    : expr :=
  Let_static code_bindings
    (Let first_made Value (Set_of_closures first [])
       (Let second_made Value (Set_of_closures second [])
          (Let_cont_non_recursive returned
             (Handler [result] (make_handler_metadata [Value])
                (Apply_cont (Apply_cont_expr done [exits_with] None)))
             (Apply
                (Apply_expr (Var first_made) kind
                   [Const (Tagged_immediate 0)] returned failed []
                   let_copied_arity let_copied_arity))))).

Definition let_copied_before (exits_with : simple) : expr :=
  let_copied_unit [Bind_code old let_copied_code] old old (Direct old)
    exits_with.

(* The copy for the first [Let]: its path is that of the body of the
   term. *)
Definition let_copied_copy : let_copy :=
  ([(old, new)], [], ([xO root], [])).

(* The code has a copy, the first [Let] makes its closure of the copy
   and the second of the old code, and the call of the first, whose
   callee is no symbol, is indirect. *)
Definition copy_let_code_one_let_statement : Prop :=
  copy_let_code let_copied_copy
    (let_copied_before (Const (Tagged_immediate 0)))
  = let_copied_unit
      [Bind_code old let_copied_code; Bind_code new let_copied_code] new old
      Indirect_unknown_arity (Const (Tagged_immediate 0)).

(* The same unit, but that it exits with the first closure: the test
   of the exits fails, and the unit is left as it is. *)
Definition copy_let_code_exit_with_closure_statement : Prop :=
  copy_let_code let_copied_copy (let_copied_before (Var first_made))
  = let_copied_before (Var first_made).

End Examples.
