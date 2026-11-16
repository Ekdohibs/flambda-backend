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

(* Induction on terms, proved once.

   [expr] is defined with three other types (handlers, static bindings
   and code), and holds lists of them. The principle that Rocq derives
   for it says nothing of what is inside a handler or inside a list, so
   a proof by induction on a term is written in the tree as a
   [Fixpoint] over the four types, with an induction of its own for
   each list. The two principles here do that once.

   - [expr_deep_ind] is for a property of terms: in the case of each
     constructor the property is known of every term directly inside
     it, the body of a handler and the body of code included, with
     [Forall] for those that are in a list.
   - [term_mutual_ind] is for four properties, one of each type.

   There is no statement here, and nothing here needs review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel.
Import ListNotations.

Definition handler_body (h : continuation_handler) : expr :=
  match h with Handler _ _ body => body end.

(* What [expr_deep_ind] knows of a static binding: the property, of the
   body of the code that it binds. *)
Definition holds_of_code (P : expr -> Prop) (b : static_binding) : Prop :=
  match b with
  | Bind_code _ c => P (code_body c)
  | Bind_deleted_code _ | Bind_symbol _ _ => True
  end.

Section Deep.
  Variable P : expr -> Prop.

  Hypothesis H_let :
    ∀ x kind d body, P body -> P (Let x kind d body).
  Hypothesis H_let_static :
    ∀ bindings body,
      Forall (holds_of_code P) bindings -> P body ->
      P (Let_static bindings body).
  Hypothesis H_let_cont :
    ∀ k params metadata handler body,
      P handler -> P body ->
      P (Let_cont_non_recursive k (Handler params metadata handler) body).
  Hypothesis H_let_cont_recursive :
    ∀ handlers body,
      Forall (fun kh => P (handler_body (snd kh))) handlers -> P body ->
      P (Let_cont_recursive handlers body).
  Hypothesis H_apply : ∀ ap, P (Apply ap).
  Hypothesis H_apply_cont : ∀ ac, P (Apply_cont ac).
  Hypothesis H_switch : ∀ scrutinee arms, P (Switch scrutinee arms).
  Hypothesis H_invalid : P Invalid.

  Fixpoint expr_deep_ind (e : expr) : P e :=
    match e with
    | Let x kind d body => H_let x kind d body (expr_deep_ind body)
    | Let_static bindings body =>
      H_let_static bindings body
        ((fix bindings_ind (bs : list static_binding)
            : Forall (holds_of_code P) bs :=
            match bs with
            | [] => Forall_nil _
            | b :: bs =>
              Forall_cons b
                (match b return holds_of_code P b with
                 | Bind_code _ (Code _ _ _ _ _ body) => expr_deep_ind body
                 | Bind_deleted_code _ | Bind_symbol _ _ => I
                 end)
                (bindings_ind bs)
            end) bindings)
        (expr_deep_ind body)
    | Let_cont_non_recursive k (Handler params metadata handler) body =>
      H_let_cont k params metadata handler body (expr_deep_ind handler)
        (expr_deep_ind body)
    | Let_cont_recursive handlers body =>
      H_let_cont_recursive handlers body
        ((fix handlers_ind (hs : list (continuation * continuation_handler))
            : Forall (fun kh => P (handler_body (snd kh))) hs :=
            match hs with
            | [] => Forall_nil _
            | (k, Handler params metadata handler) :: hs =>
              Forall_cons (k, Handler params metadata handler)
                (expr_deep_ind handler) (handlers_ind hs)
            end) handlers)
        (expr_deep_ind body)
    | Apply ap => H_apply ap
    | Apply_cont ac => H_apply_cont ac
    | Switch scrutinee arms => H_switch scrutinee arms
    | Invalid => H_invalid
    end.
End Deep.

Section Mutual.
  Variables (P : expr -> Prop) (Q : continuation_handler -> Prop)
    (S : static_binding -> Prop) (C : code -> Prop).

  Hypothesis H_let :
    ∀ x kind d body, P body -> P (Let x kind d body).
  Hypothesis H_let_static :
    ∀ bindings body,
      Forall S bindings -> P body -> P (Let_static bindings body).
  Hypothesis H_let_cont :
    ∀ k h body, Q h -> P body -> P (Let_cont_non_recursive k h body).
  Hypothesis H_let_cont_recursive :
    ∀ handlers body,
      Forall (fun kh => Q (snd kh)) handlers -> P body ->
      P (Let_cont_recursive handlers body).
  Hypothesis H_apply : ∀ ap, P (Apply ap).
  Hypothesis H_apply_cont : ∀ ac, P (Apply_cont ac).
  Hypothesis H_switch : ∀ scrutinee arms, P (Switch scrutinee arms).
  Hypothesis H_invalid : P Invalid.
  Hypothesis H_handler :
    ∀ params metadata body, P body -> Q (Handler params metadata body).
  Hypothesis H_bind_code : ∀ id c, C c -> S (Bind_code id c).
  Hypothesis H_bind_deleted_code : ∀ id, S (Bind_deleted_code id).
  Hypothesis H_bind_symbol : ∀ s const, S (Bind_symbol s const).
  Hypothesis H_code :
    ∀ params my_closure return_continuation exn_continuation metadata body,
      P body ->
      C (Code params my_closure return_continuation exn_continuation
           metadata body).

  Fixpoint expr_mutual_ind (e : expr) : P e :=
    match e with
    | Let x kind d body => H_let x kind d body (expr_mutual_ind body)
    | Let_static bindings body =>
      H_let_static bindings body
        ((fix bindings_ind (bs : list static_binding) : Forall S bs :=
            match bs with
            | [] => Forall_nil _
            | b :: bs =>
              Forall_cons b (static_binding_mutual_ind b) (bindings_ind bs)
            end) bindings)
        (expr_mutual_ind body)
    | Let_cont_non_recursive k h body =>
      H_let_cont k h body (handler_mutual_ind h) (expr_mutual_ind body)
    | Let_cont_recursive handlers body =>
      H_let_cont_recursive handlers body
        ((fix handlers_ind (hs : list (continuation * continuation_handler))
            : Forall (fun kh => Q (snd kh)) hs :=
            match hs with
            | [] => Forall_nil _
            | kh :: hs =>
              Forall_cons kh (handler_mutual_ind (snd kh)) (handlers_ind hs)
            end) handlers)
        (expr_mutual_ind body)
    | Apply ap => H_apply ap
    | Apply_cont ac => H_apply_cont ac
    | Switch scrutinee arms => H_switch scrutinee arms
    | Invalid => H_invalid
    end
  with handler_mutual_ind (h : continuation_handler) : Q h :=
    match h with
    | Handler params metadata body =>
      H_handler params metadata body (expr_mutual_ind body)
    end
  with static_binding_mutual_ind (b : static_binding) : S b :=
    match b with
    | Bind_code id c => H_bind_code id c (code_mutual_ind c)
    | Bind_deleted_code id => H_bind_deleted_code id
    | Bind_symbol s const => H_bind_symbol s const
    end
  with code_mutual_ind (c : code) : C c :=
    match c with
    | Code params my_closure return_continuation exn_continuation metadata
        body =>
      H_code params my_closure return_continuation exn_continuation metadata
        body (expr_mutual_ind body)
    end.

  Theorem term_mutual_ind :
    (∀ e, P e) ∧ (∀ h, Q h) ∧ (∀ b, S b) ∧ (∀ c, C c).
  Proof.
    exact (conj expr_mutual_ind
             (conj handler_mutual_ind
                (conj static_binding_mutual_ind code_mutual_ind))).
  Qed.
End Mutual.
