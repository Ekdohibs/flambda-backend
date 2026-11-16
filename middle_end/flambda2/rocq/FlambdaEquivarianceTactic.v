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

(* A tactic that proves, from its definition, that a function on terms
   commutes with renaming.

   The goal is an equation whose left-hand side is the function at
   renamed arguments, or a definition that unfolds to one. The tactic
   [equivariance]
   - goes by induction on the argument that the function is defined by
     recursion on ([expr_deep_ind] for a term), or unfolds the function
     if it is not recursive;
   - then, in each case, rewrites with the laws of the database
     [equivariance] and with the hypotheses of the induction, computes,
     goes under the functions of lists ([map], [forallb], [existsb],
     [flat_map], [filter], [fold_right], [fold_left]) to the elements
     of the list, and goes by cases on what the function goes by cases
     on;
   - where a law or a hypothesis does not apply as it is written, uses
     a hypothesis from right to left and shows the arguments equal;
     unfolds a renaming that is defined as a function of lists, and a
     function that is defined as an application; and makes one more
     induction, on a list that a recursive function is stuck at;
   - where a function that the function calls is at something renamed
     and no law says what it is there, states the law of that function
     from its type, proves it from its definition in the same way, and
     uses it (the last section).
   It leaves the goals that it does not close.

   So the laws of the functions that a function calls need not be
   stated: the law of a pass is proved from the definitions alone. A
   law that is stated and added to the database with [Hint Rewrite]
   is not derived again by the proofs that come after it: do that for
   a function that many others call, or whose law another file uses
   by name. A law is used where the goal has its left-hand side as it
   is written: state a law with the forms that computation gives (for
   a list of handlers renamed, [map (fun '(k, h) => ..) handlers]),
   or with a renaming that the tactic can unfold.

   A function that takes a function and is not one of those above
   needs two lemmas (it at a mapped list; it at two functions that
   agree on the elements of the list) and a rule that uses them, given
   to the tactic by redefining [eqv_extra]: see [first_answer] in
   FlambdaUnaliasParamsNamesProofs.v. Redefine it with [#[local]]: a
   redefinition replaces the one before it, for every file that comes
   after.

   A type that does not say how it is renamed (a map keyed by names)
   is declared by an instance of [Renaming_of] (the section before
   the last); without it the tactic keeps such an argument as it is,
   tries laws that are false, and is slow to fail.

   The statement of a law need not be written: [equivariance_of f] is
   the law of renaming of [f], read from its type (the section before
   the last).

   There is no statement here, and nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaProofs FlambdaEquivarianceSpec FlambdaEquivarianceProofs
  FlambdaTermInductionProofs.
From Ltac2 Require Ltac2.
Import ListNotations.
(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma forallb_of_map {A B : Type} (f : B -> bool) (g : A -> B)
    (l : list A) :
  forallb f (map g l) = forallb (fun a => f (g a)) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn. rewrite IH.
  reflexivity.
Qed.

Lemma existsb_of_map {A B : Type} (f : B -> bool) (g : A -> B)
    (l : list A) :
  existsb f (map g l) = existsb (fun a => f (g a)) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn. rewrite IH.
  reflexivity.
Qed.

Lemma forallb_ext_among {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, In a l -> f a = g a) -> forallb f l = forallb g l.
Proof.
  induction l as [ | a l IH]; intros Hfg; [ reflexivity | ]. cbn.
  rewrite (Hfg a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros b Hb. apply Hfg. right. exact Hb.
Qed.

Lemma existsb_ext_among {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, In a l -> f a = g a) -> existsb f l = existsb g l.
Proof.
  induction l as [ | a l IH]; intros Hfg; [ reflexivity | ]. cbn.
  rewrite (Hfg a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros b Hb. apply Hfg. right. exact Hb.
Qed.

Lemma flat_map_of_map {A B C : Type} (f : B -> list C) (g : A -> B)
    (l : list A) :
  flat_map f (map g l) = flat_map (fun a => f (g a)) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn. rewrite IH.
  reflexivity.
Qed.

Lemma map_of_flat_map {A B C : Type} (g : B -> C) (f : A -> list B)
    (l : list A) :
  map g (flat_map f l) = flat_map (fun a => map g (f a)) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn.
  rewrite map_app, IH. reflexivity.
Qed.

Lemma flat_map_ext_among {A B : Type} (f g : A -> list B) (l : list A) :
  (∀ a : A, In a l -> f a = g a) -> flat_map f l = flat_map g l.
Proof.
  induction l as [ | a l IH]; intros Hfg; [ reflexivity | ]. cbn.
  rewrite (Hfg a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros b Hb. apply Hfg. right. exact Hb.
Qed.

Lemma filter_of_map {A B : Type} (f : B -> bool) (g : A -> B)
    (l : list A) :
  filter f (map g l) = map g (filter (fun a => f (g a)) l).
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn. rewrite IH.
  destruct (f (g a)); reflexivity.
Qed.

Lemma filter_ext_among {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, In a l -> f a = g a) -> filter f l = filter g l.
Proof.
  induction l as [ | a l IH]; intros Hfg; [ reflexivity | ]. cbn.
  rewrite (Hfg a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros b Hb. apply Hfg. right. exact Hb.
Qed.

Lemma rev_of_map {A B : Type} (f : A -> B) (l : list A) :
  rev (map f l) = map f (rev l).
Proof. symmetry. apply map_rev. Qed.

Lemma option_map_same {A : Type} (o : option A) :
  option_map (fun a => a) o = o.
Proof. destruct o; reflexivity. Qed.

Lemma map_same {A : Type} (l : list A) : map (fun a => a) l = l.
Proof. apply map_id. Qed.

Lemma fold_right_of_map {A B C : Type} (f : B -> C -> C) (g : A -> B)
    (c : C) (l : list A) :
  fold_right f c (map g l) = fold_right (fun a acc => f (g a) acc) c l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn. rewrite IH.
  reflexivity.
Qed.

Lemma fold_right_natural {A C C' : Type} (h : C -> C')
    (f' : A -> C' -> C') (f : A -> C -> C) (c : C) (l : list A) :
  (∀ (a : A) (acc : C), In a l -> f' a (h acc) = h (f a acc)) ->
  fold_right f' (h c) l = h (fold_right f c l).
Proof.
  induction l as [ | a l IH]; intros Hf; [ reflexivity | ]. cbn.
  rewrite IH; [ apply Hf; left; reflexivity | ].
  intros b acc Hb. apply Hf. right. exact Hb.
Qed.

Lemma fold_right_ext_among {A C : Type} (f g : A -> C -> C) (c : C)
    (l : list A) :
  (∀ (a : A) (acc : C), In a l -> f a acc = g a acc) ->
  fold_right f c l = fold_right g c l.
Proof.
  induction l as [ | a l IH]; intros Hfg; [ reflexivity | ]. cbn.
  rewrite IH; [ apply Hfg; left; reflexivity | ].
  intros b acc Hb. apply Hfg. right. exact Hb.
Qed.

Lemma fold_left_of_map {A B C : Type} (f : C -> B -> C) (g : A -> B)
    (l : list A) :
  ∀ c : C,
    List.fold_left f (map g l) c
    = List.fold_left (fun acc a => f acc (g a)) l c.
Proof.
  induction l as [ | a l IH]; intros c; [ reflexivity | ]. cbn.
  apply IH.
Qed.

Lemma fold_left_natural {A C C' : Type} (h : C -> C')
    (f' : C' -> A -> C') (f : C -> A -> C) (l : list A) :
  (∀ (acc : C) (a : A), In a l -> f' (h acc) a = h (f acc a)) ->
  ∀ c : C, List.fold_left f' l (h c) = h (List.fold_left f l c).
Proof.
  induction l as [ | a l IH]; intros Hf c; [ reflexivity | ]. cbn.
  rewrite Hf by (left; reflexivity). apply IH.
  intros acc b Hb. apply Hf. right. exact Hb.
Qed.

Lemma fold_left_ext_among {A C : Type} (f g : C -> A -> C) (l : list A) :
  (∀ (acc : C) (a : A), In a l -> f acc a = g acc a) ->
  ∀ c : C, List.fold_left f l c = List.fold_left g l c.
Proof.
  induction l as [ | a l IH]; intros Hfg c; [ reflexivity | ]. cbn.
  rewrite Hfg by (left; reflexivity). apply IH.
  intros acc b Hb. apply Hfg. right. exact Hb.
Qed.

Lemma map_keys_as_map_bindings {K K' A : Type} (f : K -> K')
    (bindings : Keyed.finite_map K A) :
  Keyed.map_keys f bindings = Keyed.map_bindings f (fun a => a) bindings.
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* The database                                                             *)
(* ======================================================================== *)

Create Rewrite HintDb equivariance.

#[export] Hint Rewrite rename_variable_eqb rename_continuation_eqb
  among_rename_variable among_rename_continuation
  @nth_error_map @length_map @map_map @forallb_of_map @existsb_of_map
  @firstn_map @skipn_map @map_app @flat_map_of_map @map_of_flat_map
  @filter_of_map @rev_of_map @option_map_same @map_same @fold_right_of_map
  @fold_left_of_map @map_keys_as_map_bindings : equivariance.

(* Two laws of FlambdaEquivarianceProofs.v are written with the
   renaming outside the function. The database has them the other
   way, as the tactic writes a law that it derives: else, once the
   tactic had derived the law of the function, the two would rewrite
   each other without end. *)
Lemma eqv_let_params (r : renaming) (params : list variable)
    (kinds : list flambda_kind) (args : list simple) (body : expr) :
  FlambdaApplySpec.let_params (map (rename_variable r) params) kinds
    (map (rename_simple r) args) (rename_expr r body)
  = rename_expr r (FlambdaApplySpec.let_params params kinds args body).
Proof. symmetry. apply let_params_equivariant. Qed.

Lemma eqv_let_tuple_fields (r : renaming) (params : list variable)
    (size field : nat) (tuple : simple) (body : expr) :
  FlambdaTupledSpec.let_tuple_fields (map (rename_variable r) params) size
    field (rename_simple r tuple) (rename_expr r body)
  = rename_expr r
      (FlambdaTupledSpec.let_tuple_fields params size field tuple body).
Proof. symmetry. apply let_tuple_fields_equivariant. Qed.

(* The laws of the files that this one is built on: the free names,
   the conditions on binders, the lookups, and the functions of the
   machine and of the two rewrites that FlambdaEquivarianceProofs.v
   has. *)
#[export] Hint Rewrite
  exn_extra_simples_rename lookup_rename_handlers rename_binds_statics
  map_fst_rename_handlers
  apply_cont_continuations_renamed
  free_variables_of_simple_renamed free_variables_of_simples_renamed
  free_variables_of_value_slots_renamed free_variables_of_named_renamed
  free_variables_of_static_const_renamed
  free_variables_of_apply_cont_renamed free_variables_of_apply_renamed
  free_variables_of_expr_renamed free_variables_of_handler_renamed
  free_variables_of_binding_renamed free_variables_of_code_renamed
  free_continuations_of_expr_renamed free_continuations_of_handler_renamed
  free_continuations_of_binding_renamed free_continuations_of_code_renamed
  code_closed_renamed handler_code_closed_renamed binding_code_closed_renamed
  code_is_closed_renamed binders_distinct_renamed
  handler_binders_distinct_renamed binding_binders_distinct_renamed
  code_binders_distinct_renamed names_among_renamed lookup_rename_env
  find_arm_renamed rename_handlers_bind_statics lookup_rename_code
  consts_of_simples_renamed fold_named_equivariant fold_switch_equivariant
  fold_constants_equivariant_aux among_renamed
  simple_in_scope_equivariant simples_in_scope_equivariant
  value_slots_in_scope_equivariant named_in_scope_equivariant
  static_binding_in_scope_equivariant apply_cont_in_scope_equivariant
  expr_in_scope_equivariant handler_in_scope_equivariant
  var_in_simples_equivariant params_not_in_args_equivariant
  inlinable_b_equivariant tupled_inlinable_b_equivariant
  over_applied_b_equivariant split_over_application_equivariant
  codes_of_equivariant eqv_let_params inlined_body_equivariant
  eqv_let_tuple_fields inlined_tupled_call_equivariant
  inline_apply_equivariant inline_calls_equivariant_aux
  inline_calls_in_handler_equivariant inline_calls_in_binding_equivariant
  inline_calls_in_code_equivariant : equivariance.

(* The test of equality of two simples, which a definition makes by a
   decision procedure: it answers of two renamed simples what it
   answers of the two. Computation is kept from unfolding the
   procedure, so that the law applies. *)
Lemma eqv_rename_simple_injective (r : renaming) (s s' : simple) :
  rename_simple r s = rename_simple r s' -> s = s'.
Proof.
  destruct s, s'; cbn; intros H; try discriminate; try exact H.
  injection H as H. f_equal. exact (rename_variable_injective r _ _ H).
Qed.

Lemma eqv_simple_eq_dec {B : Type} (r : renaming) (s s' : simple)
    (a b : B) :
  (if simple_eq_dec (rename_simple r s) (rename_simple r s') then a else b)
  = (if simple_eq_dec s s' then a else b).
Proof.
  destruct (simple_eq_dec s s') as [-> | Hne],
    (simple_eq_dec (rename_simple r _) (rename_simple r _)) as [Heq | Hne'];
    try reflexivity; [ contradiction | ].
  elim Hne. exact (eqv_rename_simple_injective r _ _ Heq).
Qed.
#[export] Hint Rewrite @eqv_simple_eq_dec : equivariance.

(* The same for two lists of simples. *)
Lemma eqv_rename_simples_injective (r : renaming) (l l' : list simple) :
  map (rename_simple r) l = map (rename_simple r) l' -> l = l'.
Proof.
  revert l'. induction l as [ | s l IH]; intros [ | s' l']; cbn; intros H;
    try discriminate H; [ reflexivity | ].
  injection H as Hs Hl.
  rewrite (eqv_rename_simple_injective r s s' Hs), (IH l' Hl). reflexivity.
Qed.

Lemma eqv_simples_eq_dec {B : Type} (r : renaming) (l l' : list simple)
    (a b : B) :
  (if list_eq_dec simple_eq_dec (map (rename_simple r) l)
        (map (rename_simple r) l')
   then a else b)
  = (if list_eq_dec simple_eq_dec l l' then a else b).
Proof.
  destruct (list_eq_dec simple_eq_dec l l') as [-> | Hne],
    (list_eq_dec simple_eq_dec (map (rename_simple r) _)
       (map (rename_simple r) _)) as [Heq | Hne'];
    try reflexivity; [ contradiction | ].
  elim Hne. exact (eqv_rename_simples_injective r _ _ Heq).
Qed.
#[export] Hint Rewrite @eqv_simples_eq_dec : equivariance.


(* ======================================================================== *)
(* The tactic                                                               *)
(* ======================================================================== *)

(* What the tactic keeps in the context while it derives the laws of
   the functions that a function calls (the last section): the law of
   [f] is being derived; the law of [f], derived; the law of [f] could
   not be derived. *)
Definition eqv_deriving {T : Type} (f : T) : Prop := True.
Definition eqv_law {T : Type} (f : T) (P : Prop) : Prop := P.
Definition eqv_failed {T : Type} (f : T) : Prop := True.

(* A term mentions no renaming of the context. *)
Ltac eqv_no_renaming x :=
  first
    [ match goal with
      | r : renaming |- _ =>
        lazymatch x with context [r] => idtac end; fail 2
      end
    | idtac ].

(* A term mentions a renaming of the context. *)
Ltac eqv_has_renaming x :=
  match goal with
  | r : renaming |- _ => lazymatch x with context [r] => idtac end
  end.

Ltac eqv_head t :=
  lazymatch t with
  | ?f _ => eqv_head f
  | _ => t
  end.

(* A renaming of something that is not a term, at a constructor: its
   definition. *)
Ltac eqv_not_a_name h :=
  lazymatch h with
  | rename_variable => fail
  | rename_continuation => fail
  | _ => idtac
  end.

(* After an induction on a list of pairs, the pair at its head. *)
Ltac eqv_revert_all_but v :=
  repeat match goal with
         | H : ?T |- _ =>
           tryif constr_eq H v then fail
           else
             (lazymatch T with
              | eqv_deriving _ => fail
              | eqv_law _ _ => fail
              | eqv_failed _ => fail
              | _ => idtac
              end;
              let s := type of T in
              lazymatch s with
              | Prop => revert H
              | _ => first [ clear H | revert H ]
              end)
         end.

Ltac eqv_induction v :=
  eqv_revert_all_but v;
  let T := type of v in
  lazymatch T with
  | expr => induction v using expr_deep_ind
  | continuation_handler => destruct v
  | static_binding => destruct v
  | code => destruct v
  | _ => induction v
  end;
  intros;
  repeat match goal with
         | a : (_ * _)%type |- _ => destruct a
         end;
  repeat match goal with
         | H : Forall _ _ |- _ => rewrite Forall_forall in H
         end.

(* A hypothesis that is not one of the marks of the tactic, which are
   [True]: to rewrite with such a one does nothing, and Rocq warns
   that it makes a scheme for it. *)
Ltac eqv_not_true H :=
  let P := type of H in
  let P' := eval hnf in P in
  lazymatch P' with
  | True => fail
  | _ => idtac
  end.

Ltac eqv_rewrite_hyp :=
  match goal with
  | H : _ |- _ => eqv_not_true H; rewrite H by assumption
  end.

(* What is inside an element of a list. *)
Ltac eqv_open :=
  repeat match goal with
         | a : (_ * _)%type |- _ => destruct a
         | h : continuation_handler |- _ => destruct h
         | b : static_binding |- _ => destruct b
         | c : code |- _ => destruct c
         end;
  cbn in *.

Ltac eqv_enter l :=
  let a := fresh "a" in
  let Ha := fresh "Ha" in
  intros a Ha;
  repeat match goal with
         | H : (forall x, In x l -> _) |- _ => specialize (H a Ha)
         end;
  clear Ha;
  eqv_open.

Ltac eqv_enter_right l :=
  let a := fresh "a" in
  let acc := fresh "acc" in
  let Ha := fresh "Ha" in
  intros a acc Ha;
  repeat match goal with
         | H : (forall x, In x l -> _) |- _ => specialize (H a Ha)
         end;
  clear Ha;
  eqv_open.

Ltac eqv_enter_left l :=
  let a := fresh "a" in
  let acc := fresh "acc" in
  let Ha := fresh "Ha" in
  intros acc a Ha;
  repeat match goal with
         | H : (forall x, In x l -> _) |- _ => specialize (H a Ha)
         end;
  clear Ha;
  eqv_open.

Ltac eqv_ext :=
  lazymatch goal with
  | |- map _ ?l = map _ ?l => apply map_ext_in; eqv_enter l
  | |- forallb _ ?l = forallb _ ?l => apply forallb_ext_among; eqv_enter l
  | |- existsb _ ?l = existsb _ ?l => apply existsb_ext_among; eqv_enter l
  | |- map _ ?l = ?l =>
    transitivity (map (fun a => a) l);
    [ apply map_ext_in; eqv_enter l | apply map_id ]
  | |- ?l = map _ ?l =>
    transitivity (map (fun a => a) l);
    [ symmetry; apply map_id | apply map_ext_in; eqv_enter l ]
  | |- flat_map _ ?l = flat_map _ ?l => apply flat_map_ext_among; eqv_enter l
  | |- filter _ ?l = filter _ ?l => apply filter_ext_among; eqv_enter l
  | |- fold_right _ ?c ?l = fold_right _ ?c ?l =>
    apply fold_right_ext_among; eqv_enter_right l
  | |- fold_right ?f' _ ?l = ?h (fold_right ?f ?c ?l) =>
    apply (fold_right_natural h f' f c l); eqv_enter_right l
  | |- List.fold_left _ ?l ?c = List.fold_left _ ?l ?c =>
    apply fold_left_ext_among; eqv_enter_left l
  | |- List.fold_left ?f' ?l _ = ?h (List.fold_left ?f ?l ?c) =>
    apply (fold_left_natural h f' f l); eqv_enter_left l
  end.

Ltac eqv_fold :=
  match goal with
  | |- context [Handler (map (rename_variable ?r) ?p) ?m
                  (rename_expr ?r ?b)] =>
    change (Handler (map (rename_variable r) p) m (rename_expr r b))
      with (rename_handler r (Handler p m b))
  | |- context [Apply_cont_expr (rename_continuation ?r ?k)
                  (map (rename_simple ?r) ?args)
                  (option_map (rename_trap_action ?r) ?action)] =>
    change (Apply_cont_expr (rename_continuation r k)
              (map (rename_simple r) args)
              (option_map (rename_trap_action r) action))
      with (rename_apply_cont r (Apply_cont_expr k args action))
  | |- context [@cons ?A (?f ?x) (@nil _)] =>
    eqv_has_renaming f;
    change (@cons A (f x) (@nil A)) with (map f (@cons _ x (@nil _)))
  | |- context [@cons ?A (?f ?x) (map ?f ?l)] =>
    eqv_has_renaming f;
    change (@cons A (f x) (map f l)) with (map f (@cons _ x l))
  end.

Lemma eqv_app_congr {A B : Type} (f g : A -> B) (a b : A) :
  f = g -> a = b -> f a = g b.
Proof. intros -> ->. reflexivity. Qed.

(* One level of congruence: the arguments of the two sides, one by
   one, down to the common part of the two applications. *)
Ltac eqv_peel :=
  lazymatch goal with
  | |- ?f ?a = ?g ?b =>
    tryif constr_eq f g then apply (f_equal f)
    else (apply (eqv_app_congr f g a b); [ eqv_peel | ])
  end.

Ltac eqv_same_head :=
  lazymatch goal with
  | |- ?l = ?r =>
    let hl := eqv_head l in
    let hr := eqv_head r in
    constr_eq hl hr
  end.

(* Congruence where it loses nothing: under a constructor, under the
   connectives of booleans and of lists, and in the last argument of
   any function when the others are the same. *)
Ltac eqv_congr :=
  lazymatch goal with
  | |- ?f ?a = ?f ?b =>
    let T := type of a in
    lazymatch T with
    | forall _, _ => fail
    | _ => apply (f_equal f)
    end
  | |- ?l = ?r =>
    let hl := eqv_head l in
    let hr := eqv_head r in
    constr_eq hl hr;
    first
      [ is_constructor hl
      | lazymatch hl with
        | @app => idtac
        | andb => idtac
        | orb => idtac
        | negb => idtac
        end ];
    eqv_peel
  end.

(* A hypothesis of the induction used from right to left: the
   right-hand side is the function at the arguments before renaming,
   which always has the form of the hypothesis; what is left is that
   the arguments on the left are the renamed ones. *)
Ltac eqv_not_a_law H :=
  let T := type of H in
  lazymatch T with
  | eqv_law _ _ => fail
  | _ => idtac
  end.

Ltac eqv_hyp_back :=
  match goal with
  | H : _ |- _ = _ =>
    eqv_not_true H; eqv_not_a_law H;
    ((rewrite <- H by assumption)
     + (match goal with
        | r : renaming |- _ => rewrite <- (H r) by assumption
        end));
    eqv_same_head; eqv_peel
  end.

Module Inspect.
  Import Ltac2.Ltac2.

  Ltac2 rec count_lambdas (t : constr) : int * constr :=
    match Constr.Unsafe.kind t with
    | Constr.Unsafe.Lambda _ body =>
      let (n, b) := count_lambdas body in (Int.add n 1, b)
    | _ => (0, t)
    end.

  Ltac2 body_of (c : constr) : constr option :=
    match Constr.Unsafe.kind c with
    | Constr.Unsafe.Constant cst _ =>
      Control.plus
        (fun () =>
           Some
             (Std.eval_lazy
                { Std.rStrength := Std.Norm; Std.rBeta := false;
                  Std.rMatch := false; Std.rFix := false;
                  Std.rCofix := false; Std.rZeta := false;
                  Std.rDelta := false;
                  Std.rConst := [Std.ConstRef cst] }
                c))
        (fun _ => None)
    | _ => None
    end.

  (* As [Prepare.structural_argument], for a function whose recursion
     is under parameters (one defined in a section). *)
  Ltac2 structural_argument (lhs : constr) : constr option :=
    match Constr.Unsafe.kind lhs with
    | Constr.Unsafe.App f args =>
      match body_of f with
      | None => None
      | Some body =>
        let (k, b) := count_lambdas body in
        match Constr.Unsafe.kind b with
        | Constr.Unsafe.Fix structs i _ _ =>
          let n := Int.add k (Array.get structs i) in
          if Int.lt n (Array.length args) then Some (Array.get args n)
          else None
        | _ => None
        end
      end
    | _ => None
    end.

  (* The definition of the constant goes by cases on its last
     argument, and is not a recursion. *)
  Ltac2 is_by_cases (c : constr) : bool :=
    match body_of c with
    | None => false
    | Some body =>
      let (n, b) := count_lambdas body in
      if Int.equal n 0 then false
      else
        match Constr.Unsafe.kind b with
        | Constr.Unsafe.Case _ _ _ scrutinee _ =>
          match Constr.Unsafe.kind scrutinee with
          | Constr.Unsafe.Rel i => Int.equal i 1
          | _ => false
          end
        | _ => false
        end
    end.

  Ltac2 check_by_cases (c : constr) : unit :=
    if is_by_cases c then () else Control.zero (Tactic_failure None).

  Ltac2 rec strip_lambdas (t : constr) : constr :=
    match Constr.Unsafe.kind t with
    | Constr.Unsafe.Lambda _ body => strip_lambdas body
    | _ => t
    end.

  (* The definition of the constant is an application of a constant
     (as [map f l]): not a recursion, not a case analysis, not a
     constructor. *)
  Ltac2 is_wrapper (c : constr) : bool :=
    match Constr.Unsafe.kind c with
    | Constr.Unsafe.Constant cst _ =>
      let body :=
        Control.plus
          (fun () =>
             Some
               (Std.eval_lazy
                  { Std.rStrength := Std.Norm; Std.rBeta := false;
                    Std.rMatch := false; Std.rFix := false;
                    Std.rCofix := false; Std.rZeta := false;
                    Std.rDelta := false;
                    Std.rConst := [Std.ConstRef cst] }
                  c))
          (fun _ => None)
      in
      match body with
      | None => false
      | Some body =>
        match Constr.Unsafe.kind (strip_lambdas body) with
        | Constr.Unsafe.App f _ =>
          match Constr.Unsafe.kind f with
          | Constr.Unsafe.Constant _ _ => true
          | _ => false
          end
        | _ => false
        end
      end
    | _ => false
    end.

  Ltac2 check_wrapper (c : constr) : unit :=
    if is_wrapper c then () else Control.zero (Tactic_failure None).
End Inspect.

Ltac eqv_is_wrapper c :=
  let f := ltac2:(c |- Inspect.check_wrapper (Option.get (Ltac1.to_constr c)))
  in
  f c.

Ltac eqv_is_by_cases c :=
  let f :=
    ltac2:(c |- Inspect.check_by_cases (Option.get (Ltac1.to_constr c)))
  in
  f c.

Ltac eqv_unfold_at_constructor :=
  match goal with
  | |- context [?t ?x] =>
    lazymatch t with
    | ?c ?r =>
      let T := type of r in
      constr_eq T renaming;
      let h := eqv_head c in
      is_const h;
      eqv_not_a_name h;
      let hx := eqv_head x in
      is_constructor hx;
      let t' := eval cbv beta delta [h] in (t x) in
      progress change (t x) with t'
    end
  end.

(* A renaming that is defined as a function of lists at a renaming of
   the elements (as [map f l]): its definition, where it is at the head
   of a side of the goal; the one that a function defined by recursion
   is at is unfolded by [eqv_unfold_stuck]. Elsewhere it stays as it
   is written, so that the laws stated of it apply. *)
Ltac eqv_unfold_occurrence tx :=
  lazymatch tx with
  | ?t ?x =>
    lazymatch t with
    | ?c ?r =>
      let T := type of r in
      constr_eq T renaming;
      let h := eqv_head c in
      is_const h;
      eqv_not_a_name h;
      eqv_is_wrapper h;
      let t' := eval cbv beta delta [h] in tx in
      progress change tx with t'
    end
  end.

Ltac eqv_unfold_wrapper :=
  first
    [ lazymatch goal with |- ?l = _ => eqv_unfold_occurrence l end
    | lazymatch goal with |- _ = ?r => eqv_unfold_occurrence r end ].

(* The function at the head of the two sides, if it is defined as an
   application: its definition. *)
Ltac eqv_unfold_head :=
  lazymatch goal with
  | |- ?l = _ =>
    let h := eqv_head l in
    is_const h;
    eqv_not_a_name h;
    eqv_is_wrapper h;
    progress unfold h
  end.

(* The cases of what the function goes by cases on: a variable first,
   then an innermost scrutinee. *)
Ltac eqv_destruct_scrutinee :=
  first
    [ match goal with
      | |- context [match ?x with _ => _ end] =>
        is_var x; eqv_no_renaming x; destruct x
      end
    | match goal with
      | |- context [match ?x with _ => _ end] =>
        eqv_no_renaming x;
        lazymatch x with
        | context [match _ with _ => _ end] => fail
        | _ => idtac
        end;
        destruct x
      end
    | match goal with
      | |- context [match ?x with _ => _ end] =>
        eqv_no_renaming x; destruct x
      end ].

Ltac eqv_destruct_renamed :=
  match goal with
  | |- context [rename_handler _ ?h] => is_var h; destruct h
  | |- context [rename_apply_cont _ ?ac] => is_var ac; destruct ac
  | |- context [rename_apply _ ?ap] => is_var ap; destruct ap
  | |- context [rename_trap_action _ ?a] => is_var a; destruct a
  | |- context [rename_simple _ ?s] => is_var s; destruct s
  | |- context [rename_named _ ?d] => is_var d; destruct d
  | |- context [rename_code _ ?c] => is_var c; destruct c
  | |- context [rename_static_binding _ ?b] => is_var b; destruct b
  | |- context [rename_static_const _ ?c] => is_var c; destruct c
  end.

(* A function defined by cases on its last argument, at a variable:
   the cases of the variable. *)
Ltac eqv_destruct_argument :=
  match goal with
  | |- context [?t ?v] =>
    is_var v;
    let T := type of v in
    lazymatch T with
    | renaming => fail
    | _ => idtac
    end;
    let h := eqv_head t in
    is_const h;
    eqv_is_by_cases h;
    destruct v
  end.

(* An induction inside the proof: a function defined by recursion is
   at the image of a variable that is not a term (a list mapped, as a
   rule), and no law of the database says what it is there. Once in
   each case. *)
Definition eqv_marker : Prop := True.

Module Nested.
  Import Ltac2.Ltac2.

  Ltac2 variable_under (arg : constr) : constr option :=
    if Constr.is_var arg then None
    else
      match Constr.Unsafe.kind arg with
      | Constr.Unsafe.App _ args =>
        let n := Array.length args in
        if Int.gt n 0 then
          let last := Array.get args (Int.sub n 1) in
          if Constr.is_var last then Some last else None
        else None
      | _ => None
      end.

  Ltac2 rec first_some (f : constr -> constr option) (l : constr list)
      : constr option :=
    match l with
    | [] => None
    | t :: l =>
      match f t with
      | Some v => Some v
      | None => first_some f l
      end
    end.

  (* The term mentions a renaming of the context. *)
  Ltac2 mentions_renaming (t : constr) : bool :=
    Control.plus
      (fun () =>
         ltac1:(a |- eqv_has_renaming a) (Ltac1.of_constr t); true)
      (fun _ => false).

  Ltac2 rec find_stuck (t : constr) : constr option :=
    match Constr.Unsafe.kind t with
    | Constr.Unsafe.App _ args =>
      let here :=
        match Inspect.structural_argument t with
        | Some arg =>
          if mentions_renaming arg then variable_under arg else None
        | None => None
        end
      in
      match here with
      | Some v => Some v
      | None => first_some find_stuck (Array.to_list args)
      end
    | Constr.Unsafe.Case _ _ _ scrutinee _ => find_stuck scrutinee
    | _ => None
    end.

  Ltac2 nested () :=
    lazy_match! goal with
    | [ |- ?lhs = _ ] =>
      match find_stuck lhs with
      | Some v =>
        ltac1:(v |- let T := type of v in
                    lazymatch T with
                    | expr => fail
                    | _ => eqv_induction v
                    end)
          (Ltac1.of_constr v)
      | None => Control.zero (Tactic_failure None)
      end
    end.

  (* The argument itself, where it is not a variable. *)
  Ltac2 rec find_stuck_argument (t : constr) : constr option :=
    match Constr.Unsafe.kind t with
    | Constr.Unsafe.App _ args =>
      let here :=
        match Inspect.structural_argument t with
        | Some arg =>
          if mentions_renaming arg then
            match variable_under arg with
            | Some _ => Some arg
            | None => None
            end
          else None
        | None => None
        end
      in
      match here with
      | Some a => Some a
      | None => first_some find_stuck_argument (Array.to_list args)
      end
    | Constr.Unsafe.Case _ _ _ scrutinee _ => find_stuck_argument scrutinee
    | _ => None
    end.

  Ltac2 unfold_stuck () :=
    lazy_match! goal with
    | [ |- ?lhs = _ ] =>
      match find_stuck_argument lhs with
      | Some a => ltac1:(a |- eqv_unfold_occurrence a) (Ltac1.of_constr a)
      | None => Control.zero (Tactic_failure None)
      end
    end.

  (* No [fix] is left bare in the goal. *)
  Ltac2 rec has_fix (t : constr) : bool :=
    match Constr.Unsafe.kind t with
    | Constr.Unsafe.Fix _ _ _ _ => true
    | _ =>
      let found := { contents := false } in
      Constr.Unsafe.iter
        (fun s => if has_fix s then found.(contents) := true else ()) t;
      found.(contents)
    end.

  Ltac2 no_fix () :=
    if has_fix (Control.goal ()) then Control.zero (Tactic_failure None)
    else ().
End Nested.

Ltac eqv_unfold_stuck := ltac2:(Nested.unfold_stuck ()).

Ltac eqv_nested :=
  lazymatch goal with
  | _ : eqv_marker |- _ => fail
  | |- _ =>
    ltac2:(Nested.nested ());
    let H := fresh "Hnested" in
    assert (H : eqv_marker) by exact I
  end.

(* Computation, refused if it leaves a recursive function unfolded
   and not applied to a constructor. *)
(* Computation does not unfold the decision on lists, nor a test that
   has a law in the database, so that their laws apply. *)
Ltac eqv_folded tac :=
  with_strategy opaque
    [simple_eq_dec list_eq_dec FlambdaNormaliseSpec.var_in_simples] tac.

Ltac eqv_compute :=
  first
    [ eqv_folded
        ltac:(progress
                cbn -[Keyed.map_bindings Keyed.map_keys Keyed.map_values]);
      ltac2:(Nested.no_fix ())
    | eqv_folded ltac:(progress simpl); ltac2:(Nested.no_fix ()) ].

(* What a law of the database asks, when it asks something: a goal
   that is not yet an equation. *)
Ltac eqv_intro :=
  lazymatch goal with
  | |- forall _, _ => intro
  end.

(* A step for the functions of one file: redefined there with [:=]. *)
Ltac eqv_extra := fail.

(* A law that the tactic has derived: as it is written, at a handler
   or an application that computation has unfolded, or from right to
   left with the arguments shown equal. *)
Ltac eqv_use_law :=
  match goal with
  | H : eqv_law _ ?S |- _ =>
    first
      [ rewrite (H : S)
      | eqv_fold; rewrite (H : S)
      | rewrite <- (H : S); eqv_same_head; eqv_peel ]
  end.

(* The two steps of the last section, which need the statement of a
   law from the type of its function and the tactic itself: they are
   defined there. *)
Ltac eqv_derive := fail.
Ltac eqv_unfold_underived := fail.

(* Two lists appended, on both sides: part by part, before any
   induction is tried on a list that one of the parts maps. *)
Ltac eqv_append :=
  lazymatch goal with
  | |- @app _ _ _ = @app _ _ _ => eqv_peel
  | |- andb _ _ = andb _ _ => eqv_peel
  end.

(* What is under the [option_map]s that a term begins with. *)
Ltac eqv_under_option_map t :=
  lazymatch t with
  | option_map _ ?u => eqv_under_option_map u
  | _ => t
  end.

(* A test that the two sides share and that has no renaming in it: by
   its cases, before a law is derived for what is under it. An answer
   that may be absent, with no renaming in it, that the two sides map
   over: by its cases. *)
Ltac eqv_shared :=
  match goal with
  | |- (if ?c then _ else _) = ?rhs =>
    eqv_no_renaming c;
    lazymatch rhs with
    | context [if c then _ else _] => destruct c
    end
  | |- option_map ?f ?x = option_map ?g ?y =>
    let o := eqv_under_option_map x in
    let o' := eqv_under_option_map y in
    constr_eq o o';
    eqv_no_renaming o;
    destruct o
  end.

(* Two tests of every element of one list, one of them at renamed
   arguments: they agree on the elements. It needs the tactic itself
   and is defined after it. *)
Ltac eqv_same_test := fail.

(* To see the steps that a proof takes, as when it does not end,
   redefine this with [::=], as [Ltac eqv_note_step say ::= say.]: it
   is given a tactic that prints the name of the step taken. *)
Ltac eqv_note_step say := idtac.

Tactic Notation "eqv_say" string(s) := eqv_note_step ltac:(idtac s).

Ltac eqv_step :=
  first
    [ eqv_intro; eqv_say "intro"
    | progress autorewrite with equivariance; eqv_say "database"
    | eqv_use_law; eqv_say "use_law"
    | eqv_compute; eqv_say "compute"
    | eqv_rewrite_hyp; eqv_say "rewrite_hyp"
    | eqv_extra; eqv_say "extra"
    | eqv_append; eqv_say "append"
    | eqv_shared; eqv_say "shared"
    | eqv_same_test; eqv_say "same_test"
    | eqv_ext; eqv_say "ext"
    | eqv_fold; progress autorewrite with equivariance;
      eqv_say "fold, database"
    | eqv_fold; eqv_rewrite_hyp; eqv_say "fold, rewrite_hyp"
    | eqv_unfold_at_constructor; eqv_say "unfold_at_constructor"
    | eqv_hyp_back; eqv_say "hyp_back"
    | eqv_unfold_wrapper; eqv_say "unfold_wrapper"
    | eqv_unfold_stuck; eqv_say "unfold_stuck"
    | eqv_nested; eqv_say "nested"
    | eqv_derive; eqv_say "derive"
    | eqv_unfold_underived; eqv_say "unfold_underived"
    | eqv_destruct_scrutinee; eqv_say "destruct_scrutinee"
    | eqv_congr; eqv_say "congr"
    | eqv_destruct_renamed; eqv_say "destruct_renamed"
    | eqv_destruct_argument; eqv_say "destruct_argument"
    | eqv_unfold_head; eqv_say "unfold_head" ].

Ltac eqv_core := repeat first [ reflexivity | eqv_step ].

Ltac eqv_same_test ::=
  match goal with
  | |- context [forallb ?f' ?l] =>
    match goal with
    | |- context [forallb ?f l] =>
      eqv_has_renaming f'; eqv_no_renaming f;
      rewrite (forallb_ext_among f' f l);
      [ | eqv_enter l; solve [ eqv_core ] ]
    end
  end.

(* A statement with the renaming of the answer on the left is turned
   round: the laws that the tactic derives and the hypotheses of its
   inductions have the function at renamed arguments on the left. *)
Ltac eqv_orient :=
  lazymatch goal with
  | |- ?lhs = _ =>
    let h := eqv_head lhs in
    let T := type of h in
    lazymatch T with
    | renaming -> _ => symmetry
    | _ => idtac
    end
  | |- _ => idtac
  end.

(* The law of the function at the head of the goal is being proved:
   the mark that a derivation puts, put at the top too, so that the
   law of a function of the same recursion is not derived apart but
   had from the induction. *)
Ltac eqv_mark_top :=
  lazymatch goal with
  | |- ?lhs = _ =>
    let g := eqv_head lhs in
    tryif is_const g then
      lazymatch goal with
      | _ : eqv_deriving g |- _ => idtac
      | |- _ =>
        let D := fresh "Hderiving" in
        assert (D : eqv_deriving g) by exact I
      end
    else idtac
  | |- _ => idtac
  end.

(* The induction: the argument that the function goes by recursion on
   is a variable, or something done to a variable (a renaming of it,
   a list of it mapped). *)
Module Prepare.
  Import Ltac2.Ltac2.

  Ltac2 head_constant (lhs : constr) : constr option :=
    match Constr.Unsafe.kind lhs with
    | Constr.Unsafe.App f _ =>
      match Constr.Unsafe.kind f with
      | Constr.Unsafe.Constant _ _ => Some f
      | _ => None
      end
    | _ => None
    end.

  Ltac2 variable_of (arg : constr) : constr option :=
    if Constr.is_var arg then Some arg
    else
      match Constr.Unsafe.kind arg with
      | Constr.Unsafe.App _ args =>
        let n := Array.length args in
        if Int.gt n 0 then
          let last := Array.get args (Int.sub n 1) in
          if Constr.is_var last then Some last else None
        else None
      | _ => None
      end.

  Ltac2 prepare () :=
    ltac1:(intros;
           repeat (lazymatch goal with
                   | |- _ = _ => fail
                   | |- _ => progress hnf; intros
                   end);
           eqv_orient;
           eqv_mark_top);
    lazy_match! goal with
    | [ |- ?lhs = _ ] =>
      match Inspect.structural_argument lhs with
      | Some arg =>
        match variable_of arg with
        | Some v => ltac1:(v |- eqv_induction v) (Ltac1.of_constr v)
        | None => ()
        end
      | None =>
        match head_constant lhs with
        | Some f => ltac1:(f |- try unfold f) (Ltac1.of_constr f)
        | None => ()
        end
      end
    | [ |- _ ] => ()
    end.
End Prepare.

Tactic Notation "equivariance" :=
  ltac2:(Prepare.prepare ()); eqv_core.

(* ======================================================================== *)
(* The statement, from the type of the function                             *)
(* ======================================================================== *)

(* How a function behaves under renaming, argument by argument: an
   argument is renamed by an action or kept as it is; so is the
   result. [equivariant s f] is the law of renaming of [f] that [s]
   describes: for the signature
   [Renamed rename_continuation (Kept (Renamed rename_expr Same))] it
   is [∀ r k i e, f (rename_continuation r k) i (rename_expr r e)
   = f k i e]. *)
Inductive signature : Type -> Type :=
  | Gives {B : Type} (act : renaming -> B -> B) : signature B
  | Same {B : Type} : signature B
  | Renamed {A T : Type} (act : renaming -> A -> A) (rest : signature T)
    : signature (A -> T)
  | Kept {A T : Type} (rest : signature T) : signature (A -> T).

Fixpoint equivariant_at {T : Type} (s : signature T) (r : renaming)
    : T -> T -> Prop :=
  match s in signature T return T -> T -> Prop with
  | Gives act => fun f f' => f' = act r f
  | Same => fun f f' => f' = f
  | Renamed act rest =>
    fun f f' => ∀ a, equivariant_at rest r (f a) (f' (act r a))
  | Kept rest => fun f f' => ∀ a, equivariant_at rest r (f a) (f' a)
  end.

Definition equivariant {T : Type} (s : signature T) (f : T) : Prop :=
  ∀ r : renaming, equivariant_at s r f f.

(* The action of a renaming on a type, read from the type: the
   renaming of a name or of a part of a term, and by structure for a
   list, an option and a pair. It fails on a type that has no names in
   it (a number, a boolean, a kind), and on a type that it does not
   know. A type of one file whose action cannot be read from it (a map
   keyed by names is a list of pairs of positives) is declared by an
   instance of [Renaming_of], whose value is the function that renames
   it:
     #[local] Instance known_renaming : Renaming_of known := rename_known.
   The instances of several files hold together; the type is matched
   as it is written, not up to its definition. *)
Class Renaming_of (T : Type) : Type := renaming_of : renaming -> T -> T.

Ltac eqv_action_extra T :=
  let i := constr:(@renaming_of T _) in
  lazymatch i with
  | @renaming_of _ ?c =>
    let Tc := type of c in
    lazymatch Tc with
    | Renaming_of ?T' =>
      let _ := match constr:(tt) with _ => constr_eq T T' end in
      let a := eval cbv delta [c] in c in
      a
    end
  end.

Ltac eqv_action T :=
  lazymatch T with
  | variable => constr:(rename_variable)
  | continuation => constr:(rename_continuation)
  | simple => constr:(rename_simple)
  | named => constr:(rename_named)
  | expr => constr:(rename_expr)
  | continuation_handler => constr:(rename_handler)
  | apply_cont => constr:(rename_apply_cont)
  | apply_expr => constr:(rename_apply)
  | trap_action => constr:(rename_trap_action)
  | static_binding => constr:(rename_static_binding)
  | static_const => constr:(rename_static_const)
  | code => constr:(rename_code)
  | list ?A =>
    let a := eqv_action A in
    constr:(fun r : renaming => map (a r))
  | option ?A =>
    let a := eqv_action A in
    constr:(fun r : renaming => option_map (a r))
  | (?A * ?B)%type =>
    match A with
    | _ =>
      let a := eqv_action A in
      match B with
      | _ =>
        let b := eqv_action B in
        constr:(fun (r : renaming) (p : A * B) => (a r (fst p), b r (snd p)))
      | _ => constr:(fun (r : renaming) (p : A * B) => (a r (fst p), snd p))
      end
    | _ =>
      let b := eqv_action B in
      constr:(fun (r : renaming) (p : A * B) => (fst p, b r (snd p)))
    end
  | _ =>
    match T with
    | _ => eqv_action_extra T
    | _ =>
      let T' := eval hnf in T in
      let _ :=
        match constr:(tt) with
        | _ => tryif constr_eq T T' then fail 1 else idtac
        end
      in
      eqv_action T'
    end
  end.

Ltac eqv_signature T :=
  lazymatch T with
  | ?A -> ?B =>
    let rest := eqv_signature B in
    match A with
    | _ => let a := eqv_action A in constr:(@Renamed A B a rest)
    | _ => constr:(@Kept A B rest)
    end
  | _ =>
    match T with
    | _ => let a := eqv_action T in constr:(@Gives T a)
    | _ => constr:(@Same T)
    end
  end.

(* The law of renaming of a function, written from its type: every
   argument and the result renamed by the action of their type. It is
   the equation itself, with nothing of the signature left in it; read
   it with [Check]. Where the type does not say how an argument is
   renamed (a map whose keys are names is a list of pairs of
   positives), write the statement, or [equivariant s f] with the
   signature. *)
Ltac eqv_statement f :=
  let T := type of f in
  let s := eqv_signature T in
  let t :=
    eval cbv beta iota delta [equivariant equivariant_at] in (equivariant s f)
  in
  exact t.

Notation "'equivariance_of' f" := ltac:(eqv_statement f)
  (at level 10, f at level 9, only parsing).

(* ======================================================================== *)
(* The laws of the functions that a function calls, derived                 *)
(* ======================================================================== *)

(* Where the goal has a function at something renamed and nothing
   above has applied, the tactic states the law of that function from
   its type ([eqv_statement]), proves it by itself in a context that
   holds nothing else, and keeps it as a hypothesis, which
   [eqv_use_law] rewrites with. The functions that the function calls
   in turn get their laws the same way, inside that proof.

   A law is not derived for a renaming, for a function whose type has
   nothing to rename, for a function of a mutual recursion one of
   whose laws is being derived (its cases are those of the induction
   under way), or more than six deep. A law that could not be derived
   is not tried again in the same case; the function is then unfolded,
   if it is not defined by recursion.

   A law derived in one case of an induction is derived again in the
   next case that needs it: where that costs too much, state the law
   as a lemma and add it to the database. *)

Ltac eqv_statement_of f :=
  let T := type of f in
  let s := eqv_signature T in
  eval cbv beta iota delta [equivariant equivariant_at] in (equivariant s f).

Module Deriving.
  Import Ltac2.Ltac2.

  (* The bodies of the recursion that defines a constant. *)
  Ltac2 fix_bodies (c : constr) : constr array option :=
    match Inspect.body_of c with
    | None => None
    | Some body =>
      match Constr.Unsafe.kind (Inspect.strip_lambdas body) with
      | Constr.Unsafe.Fix _ _ _ bodies => Some bodies
      | _ => None
      end
    end.

  (* Two constants defined by one mutual recursion. *)
  Ltac2 same_block (c : constr) (d : constr) : bool :=
    match fix_bodies c with
    | None => false
    | Some b1 =>
      match fix_bodies d with
      | None => false
      | Some b2 =>
        if Int.equal (Array.length b1) (Array.length b2) then
          Constr.equal (Array.get b1 0) (Array.get b2 0)
        else false
      end
    end.

  (* The constant has a definition that is not a recursion. *)
  Ltac2 is_plain (c : constr) : bool :=
    match Inspect.body_of c with
    | None => false
    | Some body =>
      if Constr.equal body c then false
      else
        match Constr.Unsafe.kind (Inspect.strip_lambdas body) with
        | Constr.Unsafe.Fix _ _ _ _ => false
        | _ => true
        end
    end.

  Ltac2 check (b : bool) : unit :=
    if b then () else Control.zero (Tactic_failure None).
End Deriving.

Ltac eqv_same_block c d :=
  let f :=
    ltac2:(c d |- Deriving.check
                    (Deriving.same_block (Option.get (Ltac1.to_constr c))
                       (Option.get (Ltac1.to_constr d))))
  in
  f c d.

Ltac eqv_is_plain c :=
  let f :=
    ltac2:(c |- Deriving.check
                  (Deriving.is_plain (Option.get (Ltac1.to_constr c))))
  in
  f c.

(* The type of the function says that something of it is renamed. *)
Ltac eqv_has_action g :=
  let T := type of g in
  let s := eqv_signature T in
  lazymatch s with
  | context [@Renamed] => idtac
  end.

(* The function is not a renaming. *)
Ltac eqv_not_a_renaming g :=
  let T := type of g in
  lazymatch T with
  | renaming -> _ => fail
  | _ => idtac
  end.

(* The context of a derivation: what says which derivations are under
   way or have failed, and the laws derived, which are closed. *)
Ltac eqv_clear_context :=
  repeat match goal with
         | H : ?T |- _ =>
           lazymatch T with
           | eqv_deriving _ => fail
           | eqv_failed _ => fail
           | eqv_law _ _ => fail
           | _ => clear H
           end
         end.

Ltac eqv_depth_ok :=
  lazymatch goal with
  | _ : eqv_deriving _, _ : eqv_deriving _, _ : eqv_deriving _,
    _ : eqv_deriving _, _ : eqv_deriving _, _ : eqv_deriving _,
    _ : eqv_deriving _ |- _ => fail
  | |- _ => idtac
  end.

(* The law of a function of the recursion of [g] is being derived. *)
Ltac eqv_blocked g :=
  match goal with
  | _ : eqv_deriving ?g' |- _ => eqv_same_block g g'
  end.

(* No law of [g] is here, none has failed, and none is being derived
   of a function of its recursion. *)
Ltac eqv_to_derive g :=
  lazymatch goal with
  | _ : eqv_law g _ |- _ => fail
  | _ : eqv_failed g |- _ => fail
  | |- _ => idtac
  end;
  tryif eqv_blocked g then fail else idtac.

(* To see which laws a proof derives, redefine these with [::=] (as
   [idtac "derived" g]). *)
Ltac eqv_note_derived g := idtac.
Ltac eqv_note_failed g := idtac.

Ltac eqv_derive ::=
  eqv_depth_ok;
  match goal with
  | |- context [?f ?a] =>
    eqv_has_renaming a;
    let g := eqv_head f in
    is_const g;
    eqv_not_a_renaming g;
    eqv_to_derive g;
    eqv_has_action g;
    let S := eqv_statement_of g in
    let H := fresh "Hlaw" in
    first
      [ assert (H : eqv_law g S)
          by (eqv_clear_context;
              let D := fresh "Hderiving" in
              assert (D : eqv_deriving g) by exact I;
              unfold eqv_law;
              equivariance);
        eqv_note_derived g
      | assert (H : eqv_failed g) by exact I;
        eqv_note_failed g ]
  end.

(* A function that is not defined by recursion, at something renamed,
   whose law could not be derived: its definition. *)
Ltac eqv_unfold_underived ::=
  match goal with
  | |- context [?f ?a] =>
    eqv_has_renaming a;
    let g := eqv_head f in
    is_const g;
    eqv_not_a_renaming g;
    eqv_is_plain g;
    lazymatch goal with
    | _ : eqv_failed g |- _ => idtac
    end;
    progress unfold g
  end.
