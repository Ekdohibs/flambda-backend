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


(* A read of a parameter of a loop made a parameter: a normalisation
   that refines.

   This file is to be reviewed: [unbox_loops_knowing] and what it is
   made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement. A round
   of the first phase of the validation applies the pass to the term
   before Simplify, after
   FlambdaUnboxVariantsSpec.unbox_variants_knowing
   (FlambdaValidateSpec.normalise_expr_params_knowing).

   Simplify unboxes the parameters of a recursive continuation as it
   does those of a join (simplify/unboxing/), with two differences
   ([Simplify_let_cont_expr.simplify_single_recursive_handler]). The
   decision is taken before any application is seen: the handler is
   gone into with parameters of unknown type
   ([DE.add_parameters_with_unknown_types]), so a parameter is unboxed
   to what its kind says it holds
   ([Optimistic_unboxing_decision.make_optimistic_decision]), and the
   decision is not filtered by what the applications give nor by
   whether it pays ([Unbox_continuation_params.make_decisions] at
   [Recursive]). Each application is then given the new arguments:
   something in scope where the type of its argument shows one
   ([Already_in_scope] of [Unboxing_epa]), and otherwise a read that
   is bound just before it ([New_let_binding]). So for

     let acc = ref 0L in
     for i = 1 to n do acc := Int64.add !acc (Int64.of_int i) done

   where the front end makes the reference a parameter of the loop,
   which holds a boxed number, the loop after Simplify has the number
   for a parameter: the entry gives it the constant, and each turn
   the sum, which the term before boxes on the way round.

   The model has no subkinds, so the pass goes by what the term
   shows, as FlambdaUnboxVariantsSpec.unbox_variant_params does for a
   join. For a recursive continuation that is alone in its [Let_cont]
   ([unbox_loop_reads]), a [Let] of the handler that reads a parameter
   becomes a parameter ([unbox_loop_read]) where:

   - the [Let] is under [Let]s only ([on_spine]), so that every turn
     of the loop comes to it, or goes wrong before it, whatever the
     values;
   - the handler may take the variable of the [Let] as a parameter,
     and that variable is not free in the handler, as for a join;
   - every application of the continuation in the body of the
     [Let_cont] can be given what the read gives of its argument, by
     FlambdaUnboxParamsSpec.extend_args: a constant, or what a [Let]
     or a [Let_static] in scope shows of the box;
   - every application in what is left of the handler can be given
     the same, where one thing more is known: that the new parameter
     is the read of the parameter read. So a turn that passes the box
     on as it was given passes the new parameter on.

   The reads of one parameter are taken in the order of
   FlambdaUnboxVariantsSpec.next_read, and the parameters in the
   order in which they stand ([unbox_loop_params]): the new
   parameters come after those that the continuation had, those read
   from an earlier parameter first. The parameter read stays; where
   nothing else uses it, it is an unused parameter of the loop, which
   FlambdaDeadParamsSpec.drop_dead_params removes.

   A parameter is unboxed only if no read of it is left in the
   handler once the pass has made parameters of those that it can
   ([unbox_loop_param_all], by [reads_left]). A read that is left
   is a primitive that reads a box
   (FlambdaUnboxParamsSpec.reads_a_box) applied to the parameter
   itself, anywhere in the handler outside code where the parameter
   is not bound again ([reads_var]): under a test, under a static
   binding, or in the handler of a continuation that the handler
   binds, a recursive one included. No other use of the parameter
   is one, and neither is a read of another variable that holds the
   same box, as the parameter of a continuation that is given it
   is. Where there is a read left, the loop is left as it was for
   that parameter. The sources that this was seen on are
   loops over a list or a tree that test [Is_int] of their parameter
   and load its fields under the test: there Simplify had left the
   parameter as it was, and the loop with the test alone made a
   parameter was the loop of neither term.

   Not done, and the term is left as it is: an application that would
   need a read bound just before it (Simplify's [New_let_binding]),
   as where the loop is entered with a box that the term does not
   show, a parameter of the function for one; a read that is under a
   test in the handler; several recursive continuations in one
   [Let_cont].

   The result refines the term, given what is known of the symbols in
   scope ([unbox_loops_knowing_symbols_refines_statement], in the form
   in which the validation takes a rewrite into code and into loops,
   FlambdaUnboxInScopeSpec.refines_where_known). Why it holds. The
   loop and the loop with the read made a parameter are run together.
   An application gives the new parameter what the read, run where
   the application is, gives of the argument: the term shows that the
   argument holds no location, so that the read is the same on every
   heap, or that a [Let] in scope has made the read, which then gives
   its value again. A primitive that reads a box and has run gives
   the same wherever the heap evolves to, and the read is under
   [Let]s only: so the handler, entered with the values of the
   application, comes to the read or goes wrong before it, and binds
   there what the new parameter holds. The handler without the read,
   entered with the new parameter bound, refines it and exits no
   later; the applications in the handler are of the continuation
   itself, and the two runs go on in step. Where the read, run at the
   application, gives nothing of its kind, it gives nothing where the
   handler comes to it, and the handler goes wrong. The read is not
   taken to the head of the handler: with a heap, a read that gives
   nothing where the handler is entered may give a value under the
   [Let]s above it. It is a refinement and not an equivalence: where
   the read gives no value of its kind the term goes wrong, and the
   result, which no longer makes the read, may run on.

   That the pass commutes with renaming is stated in
   FlambdaUnboxVariantsRenameSpec.v, and that it keeps scope in
   FlambdaUnboxInScopeRenameSpec.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaBlockLoadSpec FlambdaUnboxSpec FlambdaUnboxParamsSpec
  FlambdaUnboxVariantsSpec FlambdaUnboxInScopeSpec.
Import ListNotations.

(* Whether a [Let] of the variable is under the bodies of [Let]s
   only, so that every run of the expression comes to it or goes
   wrong before it. *)
Fixpoint on_spine (x : variable) (e : expr) : bool :=
  match e with
  | Let y _ _ body => Pos.eqb x y || on_spine x body
  | Let_static _ _ | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid =>
    false
  end.

(* The recursive [Let_cont] of [k] alone with the read made a
   parameter: the applications in the body are given what the term
   shows that the read gives, and those in what is left of the
   handler the same, knowing there that the new parameter is the read
   of the parameter read. *)
Definition unbox_loop_read (known : facts) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) (r : read)
    : option (list variable * handler_metadata * expr * expr) :=
  let f := read_variable r in
  let p := read_primitive r in
  let i := read_position r in
  let* x := nth_error params i in
  if may_take params metadata f
     && negb (var_free_b f handler)
     && on_spine f handler
  then
    let* body := extend_applications known k i p body in
    let* without :=
      extend_applications
        (facts_under_let (facts_under_params known params) f
           (Prim (Unary p) [Var x]))
        k i p (read_without r)
    in
    Some (params ++ [f], with_one_more metadata (read_kind r), without, body)
  else None.

(* A loop that is being unboxed: its parameters, its metadata, its
   handler, and the body of its [Let_cont]. *)
Definition loop_state : Type :=
  (list variable * handler_metadata * expr * expr)%type.

Definition loop_of (k : continuation) (state : loop_state) : expr :=
  let '(params, metadata, handler, body) := state in
  Let_cont_recursive [(k, Handler params metadata handler)] body.

(* The reads of the parameter at position [i] made parameters one
   after the other, in the order of [next_read], as long as each can
   be. *)
Fixpoint unbox_loop_param (fuel : nat) (known : facts) (k : continuation)
    (i : nat) (state : loop_state) : loop_state :=
  match fuel with
  | 0 => state
  | S fuel =>
    let '(params, metadata, handler, body) := state in
    match
      (let* param := nth_error params i in
       let* r := next_read [param] handler in
       unbox_loop_read known k params metadata handler body
         (Read (read_variable r) (read_kind r) (read_primitive r) i
            (read_without r)))
    with
    | Some state => unbox_loop_param fuel known k i state
    | None => state
    end
  end.

(* Whether a defining expression is a read of the variable [x]: a
   primitive that reads a box (FlambdaUnboxParamsSpec.reads_a_box)
   applied to it. *)
Definition reads_var_named (x : variable) (d : named) : bool :=
  match d with
  | Prim (Unary p) [Var y] => reads_a_box p && Pos.eqb y x
  | _ => false
  end.

(* Whether an expression has a read of the variable [x], outside code
   and where [x] is not bound again: in the expression, under its
   static bindings, and in the handlers of the continuations that it
   binds, recursive ones included. *)
Fixpoint reads_var (x : variable) (e : expr) {struct e} : bool :=
  match e with
  | Let y _ d body =>
    reads_var_named x d || (negb (Pos.eqb y x) && reads_var x body)
  | Let_static _ body => reads_var x body
  | Let_cont_non_recursive _ h body =>
    handler_reads_var x h || reads_var x body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => handler_reads_var x h) handlers
    || reads_var x body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => false
  end
with handler_reads_var (x : variable) (h : continuation_handler) {struct h}
    : bool :=
  match h with
  | Handler params _ handler => negb (among x params) && reads_var x handler
  end.

(* Whether the handler still reads the parameter at position [i]. *)
Definition reads_left (i : nat) (state : loop_state) : bool :=
  let '(params, _, handler, _) := state in
  match nth_error params i with
  | Some param => reads_var param handler
  | None => false
  end.

(* The reads of the parameter at position [i] made parameters, if that
   leaves no read of it; the loop as it was, if one is left. *)
Definition unbox_loop_param_all (fuel : nat) (known : facts)
    (k : continuation) (i : nat) (state : loop_state) : loop_state :=
  let unboxed := unbox_loop_param fuel known k i state in
  if reads_left i unboxed then state else unboxed.

(* The same for each of the positions given, in turn. *)
Fixpoint unbox_loop_params (fuel : nat) (known : facts) (k : continuation)
    (positions : list nat) (state : loop_state) : loop_state :=
  match positions with
  | [] => state
  | i :: positions =>
    unbox_loop_params fuel known k positions
      (unbox_loop_param_all fuel known k i state)
  end.

Definition unbox_loop_reads (known : facts) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) : expr :=
  loop_of k
    (unbox_loop_params (lets_in handler) known k (seq 0 (length params))
       (params, metadata, handler, body)).

Fixpoint unbox_loops_knowing (known : facts) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (unbox_loops_knowing (facts_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (unbox_loops_knowing (facts_under_statics known bindings) body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata
         (unbox_loops_knowing (facts_under_params known params) handler))
      (unbox_loops_knowing known body)
  | Let_cont_recursive [(k, Handler params metadata handler)] body =>
    unbox_loop_reads known k params metadata handler
      (unbox_loops_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (unbox_loops_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The pass refines, given what is known of the symbols in scope. *)
Definition unbox_loops_knowing_symbols_refines_statement : Prop :=
  refines_where_known unbox_loops_knowing.
