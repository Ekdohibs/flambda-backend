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


(* Unboxing a parameter of a continuation that every application gives
   something known: a normalisation that refines.

   This file is to be reviewed: [unbox_known_params] and what it is made
   of, and the statement. The validation does not run this pass: it
   runs FlambdaUnboxVariantsSpec.unbox_variant_params, which is made as
   this one is and uses what is defined here (that file says how the
   two differ).

   Simplify unboxes the parameters of a continuation
   (simplify/unboxing/): where the type of a parameter says that it is
   a block of one tag and size, a variant or a boxed or tagged number,
   the continuation gets one more parameter for each field, for the tag,
   for whether the value is an immediate, or for the number, and every
   application gets the matching arguments
   ([Unbox_continuation_params.make_decisions],
   [compute_extra_params_and_args]; [Apply_cont_rewrite.make_rewrite]
   adds them, the new parameters after those the continuation had). In
   the handler the type of the old parameter is made to say that its
   fields are the new parameters ([Build_unboxing_denv.denv_of_decision]),
   so that every load of a field there becomes the new parameter; the
   old parameter goes if nothing else uses it, and with it the
   allocation at an application that made the block only to pass it.
   What an application gives for a new parameter is the field itself
   where its type knows it ([Unboxing_epa.unbox_arg]: an
   [Already_in_scope] argument), and a load made before the application
   where it does not (a [New_let_binding] of [Unboxers.Field.unboxing_prim],
   [%block_load] with the tag and the size of the decision).

   [unbox_known_params] does the first of the two, where it can be
   read off the term. At a non-recursive [Let_cont] whose handler
   begins with a [Let] of a variable [f] to a primitive that reads a
   box ([reads_a_box]: a load not marked [Mutable], [Is_int],
   [Get_tag], [Unbox_number], [Untag_immediate]) applied to a parameter
   [x], it removes that [Let] and makes [f] one more parameter, the
   last; and it gives every application of the continuation in the
   body of the [Let_cont] one more argument, the last: what the
   primitive is known to give of the argument that the application has
   for [x] ([projection_of]). That is known in four ways, the first
   three each a function of another file, where the conditions are:
   - the argument is a variable bound by a [Let] to a [Make_block], or
     a symbol bound to a static block
     (FlambdaBlockLoadSpec.load_known_named): a load gives what the
     field holds, [Is_int] no, [Get_tag] the tag;
   - the argument is a variable bound by a [Let] to a [Box_number] or
     a [Tag_immediate], or a symbol bound to a static boxed number
     (FlambdaUnboxSpec.unbox_known_named): the unboxing or the
     untagging gives what was wrapped;
   - the argument is a constant on which the primitive has a value
     that is a constant (FlambdaNormaliseSpec.fold_named): [Is_int] of
     a tagged immediate, an untagging;
   - the argument is a variable, and a [Let] in scope has bound a
     variable to the same primitive applied to it, neither having been
     bound again since ([reads_under_let], looked up by
     FlambdaCseSpec.computed_by): the argument given is that variable.
     So an application that gives a box which the term has already
     read, a parameter of the function that it has unboxed for a
     comparison, gives what was read. Simplify does the same: where it
     simplifies the read it makes the type of the box say what was
     read ([Simplify_unary_primitive.simplify_unbox_number] and the
     others, by [Simplify_common.simplify_projection]), and the
     argument for the new parameter is then in scope
     ([Unboxing_epa.unbox_arg], [Known_result]).
   Then it does the same for the [Let] that the handler now begins
   with ([unbox_heads]), so a handler that begins by reading several
   fields of a parameter, or a field of a field, has them all made
   parameters, in the order in which it reads them. The parameter that
   was read stays, with its arguments: where the handler no longer uses
   it, removing it is another pass's.

   What stops it, for one [Let] and those after it:
   - an application of the continuation whose argument is not known in
     one of the four ways. No [Let] is added at an application: the
     pass makes no name, and an arm of a [Switch] has no place for
     one. So a parameter that some application gives a value from
     elsewhere (a parameter of the function, the result of a call) is
     left boxed, where Simplify adds a load. That is also what stops
     it for a primitive that does not read a box: [projection_of]
     gives no argument for an application. The pass does not look at
     the primitive otherwise, so the [Let] at the head of a handler
     that the body never applies becomes a parameter whatever its
     unary primitive is, and nothing gives it an argument;
   - an [Apply] in the scope of the continuation that returns to it or
     raises to it: the call applies the continuation to values that
     the term does not show;
   - an exception handler, which Simplify does not unbox either;
   - parameters that are not distinct, or not as many as the kinds of
     the metadata; a [Let] of a name that is already a parameter;
   - a recursive [Let_cont]: its handlers are left as they are, and so
     are the bodies of code (FlambdaNormaliseSpec.rewrite_code takes
     the pass into code, with nothing known there). The applications
     that are inside the handlers of a recursive [Let_cont], or of any
     other continuation, in the scope of the continuation being
     unboxed are given their argument like the others.

   Scopes. An application of the same name under a binding of that
   name by another [Let_cont] is an application of that other
   continuation and is left alone ([extend_applications] stops there).
   What is known of a variable is forgotten under a binder of it, as in
   the files that say what is known. Nothing is asked of the names of
   the term.

   The result refines the term ([unbox_known_params_refines_statement]),
   under all environments. It is not claimed to be equivalent to it,
   and here is a case in which it is not: a static block may hold a
   field that is not of the kind of values, as nothing tests the
   fields of a static block when it is bound. The load of such a
   field in the handler has no value; the argument at the application
   is the field itself (FlambdaBlockLoadSpec.field_loaded gives what
   the field holds, whatever its kind), and a new parameter of the
   kind of the field takes it, so the term after runs where the term
   before is undefined. So the pass is for the term before Simplify.

   What it is not: it does not decide as Simplify decides. Simplify
   unboxes by the type of the parameter, to a bounded depth, and only
   where that is of use; this pass goes by what the handler reads
   first. Where the two differ the validation fails; it never accepts
   more. The new parameters are in the order of the [Let]s of the
   handler, and Simplify's are in the reverse of the order of the
   fields ([Continuation_extra_params_and_args.add] puts each new
   parameter first): the two agree where the source reads the fields
   last to first, as the pattern matching of the front end does.

   Renaming: the functions here compare names by equality only and bind
   none that the term has not got. FlambdaUnboxParamsRenameSpec.v has
   the statement that the pass commutes with a renaming.

   [extend_applications_by] is [extend_applications] with the new
   arguments given by a function; the pass does not use it.
   FlambdaUnboxVariantsSpec.v does, for a parameter that is read under
   a test.

   The proofs are in FlambdaUnboxParamsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaBlockLoadSpec
  FlambdaUnboxSpec FlambdaCseSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* The primitives by which a handler reads a parameter that Simplify
   unboxes: the load of a field that is not marked [Mutable], the two
   tests of a variant, and the unwrapping of a number. *)
Definition reads_a_box (p : unary_primitive) : bool :=
  match p with
  | Block_load _ (Immutable | Immutable_unique) _ | Is_int _ | Get_tag
  | Unbox_number _ | Untag_immediate =>
    true
  | Block_load _ Mutable _ | Is_null | Array_length _ | String_length
  | Opaque_identity _ | Unary_int_arith _ _ | Num_conv _ _ | Boolean_not
  | Box_number _ | Tag_immediate | Reinterpret_64_bit_word _ =>
    false
  end.

(* What is known from the term of the blocks and of the wrapped numbers
   in scope: what FlambdaBlockLoadSpec.v and FlambdaUnboxSpec.v keep;
   and the reads of boxes that the [Let]s in scope have made: the
   variables bound to a primitive that reads a box, applied to a
   variable, each with the primitive and that variable, in the form in
   which FlambdaCseSpec.v keeps what is computed. *)
Record facts : Type := Facts {
  facts_blocks : known_blocks;
  facts_numbers : known_numbers;
  facts_reads : computed
}.

Definition no_facts : facts := Facts no_blocks no_numbers [].

(* The reads known in the body of a [Let] of [x] to [d]: none that
   mentions the old [x], and that the new one is the read that [d]
   makes, if [d] is a primitive that reads a box applied to a variable
   other than [x]. *)
Definition reads_under_let (known : computed) (x : variable) (d : named)
    : computed :=
  let known := forget_computed x known in
  match d with
  | Prim (Unary p) [Var y] =>
    if reads_a_box p && negb (Pos.eqb x y)
    then (x, (Unary p, [Var y])) :: known
    else known
  | _ => known
  end.

(* What is known in the body of a [Let], in that of a [Let_static], and
   in a handler, whose parameters are binders. *)
Definition facts_under_let (known : facts) (x : variable) (d : named)
    : facts :=
  Facts (known_under_let (facts_blocks known) x d)
    (numbers_under_let (facts_numbers known) x d)
    (reads_under_let (facts_reads known) x d).

Definition facts_under_statics (known : facts)
    (bindings : list static_binding) : facts :=
  Facts (known_under_statics (facts_blocks known) bindings)
    (numbers_under_statics (facts_numbers known) bindings)
    (facts_reads known).

Definition facts_under_params (known : facts) (params : list variable)
    : facts :=
  Facts (forget_vars params (facts_blocks known))
    (forget_wrapped_vars params (facts_numbers known))
    (forget_computed_vars params (facts_reads known)).

(* ======================================================================== *)
(* What reading a box gives                                                 *)
(* ======================================================================== *)

(* What such a primitive applied to an argument is known to give, where
   the term alone says it: by a constant argument, by a known block, by
   a known wrapped number, or, where none of these says, by a [Let] in
   scope that has made the same read of the argument, whose variable is
   then given. *)
Definition projection_of (known : facts) (p : unary_primitive)
    (arg : simple) : option simple :=
  if reads_a_box p
  then
    let d := Prim (Unary p) [arg] in
    match fold_named d with
    | Simple s => Some s
    | _ =>
      match load_known_named (facts_blocks known) d with
      | Simple s => Some s
      | _ =>
        match unbox_known_named (facts_numbers known) d with
        | Simple s => Some s
        | _ =>
          option_map Var (computed_by (facts_reads known) (Unary p) [arg])
        end
      end
    end
  else None.

(* ======================================================================== *)
(* One more argument at every application                                   *)
(* ======================================================================== *)

(* The arguments of an application with one more, the last: what [p]
   gives of the argument at position [i], if that is known. *)
Definition extend_args (known : facts) (i : nat) (p : unary_primitive)
    (args : list simple) : option (list simple) :=
  let* arg := nth_error args i in
  let* projection := projection_of known p arg in
  Some (args ++ [projection]).

(* An application, with one more argument if it is of [k]. Its trap
   action stays. *)
Definition extend_apply_cont (known : facts) (k : continuation) (i : nat)
    (p : unary_primitive) (ac : apply_cont) : option apply_cont :=
  if Pos.eqb (ac_continuation ac) k
  then
    let* args := extend_args known i p (ac_args ac) in
    Some (Apply_cont_expr (ac_continuation ac) args (ac_trap_action ac))
  else Some ac.

Fixpoint extend_arms (known : facts) (k : continuation) (i : nat)
    (p : unary_primitive) (arms : list (Z * apply_cont))
    : option (list (Z * apply_cont)) :=
  match arms with
  | [] => Some []
  | (discriminant, ac) :: arms =>
    let* ac := extend_apply_cont known k i p ac in
    let* arms := extend_arms known k i p arms in
    Some ((discriminant, ac) :: arms)
  end.

(* All of them, if each is there. *)
Fixpoint all_some {A : Type} (l : list (option A)) : option (list A) :=
  match l with
  | [] => Some []
  | a :: l =>
    let* a := a in
    let* l := all_some l in
    Some (a :: l)
  end.

(* The expression with one more argument at every application of [k]
   that is in the scope of the continuation of that name which is in
   scope where the expression starts; nothing if some such application
   has an argument at position [i] of which [p] is not known, or if an
   [Apply] returns or raises to the continuation. Under a binding of the
   same name the applications are of another continuation: a handler of
   a non-recursive [Let_cont] that binds [k] is gone into, as [k] is in
   it the one outside, and its body is not; a recursive [Let_cont] one
   of whose continuations is [k] is left as it is. *)
Fixpoint extend_applications (known : facts) (k : continuation) (i : nat)
    (p : unary_primitive) (e : expr) : option expr :=
  match e with
  | Let x kind d body =>
    let* body :=
      extend_applications (facts_under_let known x d) k i p body
    in
    Some (Let x kind d body)
  | Let_static bindings body =>
    let* body :=
      extend_applications (facts_under_statics known bindings) k i p body
    in
    Some (Let_static bindings body)
  | Let_cont_non_recursive k' h body =>
    let* h := extend_in_handler known k i p h in
    if Pos.eqb k' k then Some (Let_cont_non_recursive k' h body)
    else
      let* body := extend_applications known k i p body in
      Some (Let_cont_non_recursive k' h body)
  | Let_cont_recursive handlers body =>
    if is_bound k handlers then Some e
    else
      let* handlers :=
        all_some
          (map
             (fun '(k', h) =>
                let* h := extend_in_handler known k i p h in
                Some (k', h))
             handlers)
      in
      let* body := extend_applications known k i p body in
      Some (Let_cont_recursive handlers body)
  | Apply ap =>
    if Pos.eqb (ap_continuation ap) k || Pos.eqb (ap_exn_continuation ap) k
    then None
    else Some e
  | Apply_cont ac =>
    let* ac := extend_apply_cont known k i p ac in
    Some (Apply_cont ac)
  | Switch scrutinee arms =>
    let* arms := extend_arms known k i p arms in
    Some (Switch scrutinee arms)
  | Invalid => Some e
  end
with extend_in_handler (known : facts) (k : continuation) (i : nat)
    (p : unary_primitive) (h : continuation_handler)
    : option continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let* handler :=
      extend_applications (facts_under_params known params) k i p handler
    in
    Some (Handler params metadata handler)
  end.

(* ======================================================================== *)
(* One more argument at every application, by a function of the arguments   *)
(* ======================================================================== *)

(* [extend_applications], with the new arguments of an application
   given by a function of what is known and of its arguments:
   [extend_applications known k i p] is [extend_applications_by] at
   [fun known => extend_args known i p]. It is here for
   FlambdaUnboxVariantsSpec.v, which gives the arguments otherwise. *)
Definition extend_apply_cont_by
    (extend : facts -> list simple -> option (list simple)) (known : facts)
    (k : continuation) (ac : apply_cont) : option apply_cont :=
  if Pos.eqb (ac_continuation ac) k
  then
    let* args := extend known (ac_args ac) in
    Some (Apply_cont_expr (ac_continuation ac) args (ac_trap_action ac))
  else Some ac.

Fixpoint extend_arms_by
    (extend : facts -> list simple -> option (list simple)) (known : facts)
    (k : continuation) (arms : list (Z * apply_cont))
    : option (list (Z * apply_cont)) :=
  match arms with
  | [] => Some []
  | (discriminant, ac) :: arms =>
    let* ac := extend_apply_cont_by extend known k ac in
    let* arms := extend_arms_by extend known k arms in
    Some ((discriminant, ac) :: arms)
  end.

Fixpoint extend_applications_by
    (extend : facts -> list simple -> option (list simple)) (known : facts)
    (k : continuation) (e : expr) : option expr :=
  match e with
  | Let x kind d body =>
    let* body :=
      extend_applications_by extend (facts_under_let known x d) k body
    in
    Some (Let x kind d body)
  | Let_static bindings body =>
    let* body :=
      extend_applications_by extend (facts_under_statics known bindings) k
        body
    in
    Some (Let_static bindings body)
  | Let_cont_non_recursive k' h body =>
    let* h := extend_in_handler_by extend known k h in
    if Pos.eqb k' k then Some (Let_cont_non_recursive k' h body)
    else
      let* body := extend_applications_by extend known k body in
      Some (Let_cont_non_recursive k' h body)
  | Let_cont_recursive handlers body =>
    if is_bound k handlers then Some e
    else
      let* handlers :=
        all_some
          (map
             (fun '(k', h) =>
                let* h := extend_in_handler_by extend known k h in
                Some (k', h))
             handlers)
      in
      let* body := extend_applications_by extend known k body in
      Some (Let_cont_recursive handlers body)
  | Apply ap =>
    if Pos.eqb (ap_continuation ap) k || Pos.eqb (ap_exn_continuation ap) k
    then None
    else Some e
  | Apply_cont ac =>
    let* ac := extend_apply_cont_by extend known k ac in
    Some (Apply_cont ac)
  | Switch scrutinee arms =>
    let* arms := extend_arms_by extend known k arms in
    Some (Switch scrutinee arms)
  | Invalid => Some e
  end
with extend_in_handler_by
    (extend : facts -> list simple -> option (list simple)) (known : facts)
    (k : continuation) (h : continuation_handler)
    : option continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let* handler :=
      extend_applications_by extend (facts_under_params known params) k
        handler
    in
    Some (Handler params metadata handler)
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* Where a name is in a list, the first being at 0. *)
Fixpoint position (x : variable) (xs : list variable) : option nat :=
  match xs with
  | [] => None
  | y :: xs => if Pos.eqb x y then Some 0 else option_map S (position x xs)
  end.

(* Whether a handler with these parameters may take [f] as one more: it
   is not an exception handler, its parameters are distinct and as many
   as its kinds, and [f] is not one of them. *)
Definition may_take (params : list variable) (metadata : handler_metadata)
    (f : variable) : bool :=
  negb (hm_is_exn_handler metadata)
  && distinct params
  && Nat.eqb (length (hm_params_kinds metadata)) (length params)
  && negb (among f params).

(* The metadata of a handler that takes one more parameter, of the kind
   given, after those it had. It is not an exception handler. *)
Definition with_one_more (metadata : handler_metadata) (kind : flambda_kind)
    : handler_metadata :=
  make_handler_metadata (hm_params_kinds metadata ++ [kind]).

(* The [Let_cont] of [k], with each [Let] that its handler begins with
   and that reads a box which is a parameter made a parameter, as long
   as every application in [body] can be given its argument. *)
Fixpoint unbox_heads (known : facts) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) : expr :=
  match handler with
  | Let f kind (Prim (Unary p) [Var x]) rest =>
    match
      (if may_take params metadata f then position x params else None)
    with
    | Some i =>
      match extend_applications known k i p body with
      | Some body =>
        unbox_heads known k (params ++ [f]) (with_one_more metadata kind)
          rest body
      | None => Let_cont_non_recursive k (Handler params metadata handler) body
      end
    | None => Let_cont_non_recursive k (Handler params metadata handler) body
    end
  | _ => Let_cont_non_recursive k (Handler params metadata handler) body
  end.

Fixpoint unbox_params_knowing (known : facts) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (unbox_params_knowing (facts_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings
      (unbox_params_knowing (facts_under_statics known bindings) body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    unbox_heads known k params metadata
      (unbox_params_knowing (facts_under_params known params) handler)
      (unbox_params_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (unbox_params_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition unbox_known_params (e : expr) : expr :=
  unbox_params_knowing no_facts e.

Definition unbox_known_params_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (unbox_known_params e).
