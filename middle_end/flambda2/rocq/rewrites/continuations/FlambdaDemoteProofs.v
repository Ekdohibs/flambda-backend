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

(* Proofs of the statements of FlambdaDemoteSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNormaliseProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaContAlikeProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaWellFormedSpec FlambdaTrapSpec
  FlambdaTrapProofs FlambdaEquivarianceTactic FlambdaDemoteSpec.
Import ListNotations.

(* ======================================================================== *)
(* The machine does not read the mark of a handler                          *)
(* ======================================================================== *)

(* A handler with its mark cleared is the handler: the two closures,
   applied to the same values, come to the same state
   (FlambdaContAlikeProofs.let_cont_handler_alike). *)
Lemma clear_exn_handler_equivalent (k : continuation)
    (h : continuation_handler) (body : expr) :
  expr_equivalent (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k (clear_exn_handler h) body).
Proof.
  intros vars conts statics hp rd. apply let_cont_handler_alike.
  intros vs statics' hp' rd'. destruct h as [params metadata handler]. simpl.
  apply result_related_refl. exact equivalent_refl.
Qed.

(* ======================================================================== *)
(* The machine does not execute a trap action                               *)
(* ======================================================================== *)

Lemma expr_equivalent_same (e : expr) : expr_equivalent e e.
Proof. intros vars conts statics hp rd. apply equivalent_refl. Qed.

Lemma apply_cont_stripped (k : continuation) (ac : apply_cont) :
  expr_equivalent (Apply_cont ac) (Apply_cont (strip_trap k ac)).
Proof.
  intros vars conts statics hp rd. apply step_eq_equivalent.
  rewrite !step_apply_cont. unfold strip_trap.
  destruct (traps k ac); reflexivity.
Qed.

Lemma find_arm_stripped (k : continuation) (i : Z)
    (arms : list (Z * apply_cont)) :
  find_arm i (map (fun arm => (fst arm, strip_trap k (snd arm))) arms)
  = option_map (strip_trap k) (find_arm i arms).
Proof.
  induction arms as [ | [discriminant ac] arms IH]; simpl; [ reflexivity | ].
  destruct (Z.eqb i discriminant); [ reflexivity | exact IH ].
Qed.

Lemma switch_stripped (k : continuation) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  expr_equivalent (Switch scrutinee arms)
    (Switch scrutinee
       (map (fun arm => (fst arm, strip_trap k (snd arm))) arms)).
Proof.
  intros vars conts statics hp rd.
  pose proof (step_switch scrutinee arms vars conts statics) as Hstep.
  pose proof
    (step_switch scrutinee
       (map (fun arm => (fst arm, strip_trap k (snd arm))) arms) vars conts
       statics) as Hstep'.
  destruct (eval_simple vars scrutinee) as [v | ];
    [ | apply step_eq_equivalent; congruence ].
  destruct (naked_immediate_of_value v) as [i | ];
    [ | apply step_eq_equivalent; congruence ].
  rewrite find_arm_stripped in Hstep'.
  destruct (find_arm i arms) as [ac | ]; simpl in Hstep';
    [ | apply step_eq_equivalent; congruence ].
  exact
    (equivalent_step _ _ _ _ (Hstep hp rd_) (Hstep' hp rd_)
       (apply_cont_stripped k ac vars conts statics hp rd_)).
Qed.

Fixpoint strip_traps_binds_statics (k : continuation) (e : expr) :
  binds_statics (strip_traps k e) = binds_statics e
with strip_traps_handler_binds_statics (k : continuation)
    (h : continuation_handler) :
  handler_binds_statics (strip_traps_handler k h) = handler_binds_statics h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply strip_traps_binds_statics.
    + rewrite (strip_traps_binds_statics k body). reflexivity.
    + rewrite (strip_traps_handler_binds_statics k h),
        (strip_traps_binds_statics k body).
      reflexivity.
    + rewrite (strip_traps_binds_statics k body). f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (strip_traps_handler_binds_statics k h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply strip_traps_binds_statics.
Qed.

Definition handler_stripped (k : continuation) (h : continuation_handler)
    : Prop :=
  match h with
  | Handler _ _ handler => expr_equivalent handler (strip_traps k handler)
  end.

(* The handlers of a recursive [Let_cont], one after the other. *)
Lemma let_cont_recursive_handlers_stripped (k : continuation) (body : expr)
    (todo : list (continuation * continuation_handler)) :
  Forall (fun kh => handler_stripped k (snd kh)) todo ->
  ∀ done : list (continuation * continuation_handler),
    expr_equivalent (Let_cont_recursive (done ++ todo) body)
      (Let_cont_recursive
         (done
          ++ map (fun kh => (fst kh, strip_traps_handler k (snd kh))) todo)
         body).
Proof.
  induction 1 as [ | [k' h] todo Hh _ IH]; intros done;
    [ apply expr_equivalent_same | ].
  cbn [map fst snd].
  destruct h as [params metadata handler]. cbn [snd handler_stripped] in Hh.
  eapply expr_equivalent_trans.
  - exact
      (plug_equivalent
         (Let_cont_recursive_handler done k' params metadata Hole todo body)
         handler (strip_traps k handler)
         (fun _ => eq_sym (strip_traps_binds_statics k handler)) Hh).
  - specialize
      (IH
         (done
          ++ [(k', strip_traps_handler k (Handler params metadata handler))])).
    rewrite <- !app_assoc in IH. exact IH.
Qed.

Fixpoint strip_traps_equivalent (k : continuation) (e : expr) :
  expr_equivalent e (strip_traps k e)
with strip_traps_handler_equivalent (k : continuation)
    (h : continuation_handler) : handler_stripped k h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [strip_traps].
    + exact
        (plug_equivalent (Let_body x kind d Hole) body (strip_traps k body)
           (fun H => False_ind _ (Bool.diff_false_true H))
           (strip_traps_equivalent k body)).
    + exact
        (plug_equivalent (Let_static_body bindings Hole) body
           (strip_traps k body)
           (fun H => False_ind _ (Bool.diff_false_true H))
           (strip_traps_equivalent k body)).
    + pose proof (strip_traps_handler_equivalent k h) as Hh.
      destruct h as [params metadata handler]. cbn [handler_stripped] in Hh.
      eapply expr_equivalent_trans.
      * exact
          (plug_equivalent
             (Let_cont_non_recursive_body k'
                (Handler params metadata handler) Hole)
             body (strip_traps k body)
             (fun H => False_ind _ (Bool.diff_false_true H))
             (strip_traps_equivalent k body)).
      * exact
          (plug_equivalent
             (Let_cont_non_recursive_handler k' params metadata Hole
                (strip_traps k body))
             handler (strip_traps k handler)
             (fun H => False_ind _ (Bool.diff_false_true H)) Hh).
    + eapply expr_equivalent_trans.
      * exact
          (plug_equivalent (Let_cont_recursive_body handlers Hole) body
             (strip_traps k body)
             (fun H => False_ind _ (Bool.diff_false_true H))
             (strip_traps_equivalent k body)).
      * apply (let_cont_recursive_handlers_stripped k (strip_traps k body)
                 handlers
                 ltac:(induction handlers as [ | [k' h] handlers IH];
                       constructor;
                       [ apply strip_traps_handler_equivalent | exact IH ])
                 []).
    + apply expr_equivalent_same.
    + apply apply_cont_stripped.
    + apply switch_stripped.
    + apply expr_equivalent_same.
  - destruct h as [params metadata handler]. cbn [handler_stripped].
    apply strip_traps_equivalent.
Qed.

(* ======================================================================== *)
(* The rewrite is an equivalence                                            *)
(* ======================================================================== *)

Lemma demote_handlers_binding_binds (binding : static_binding) :
  binding_binds (demote_handlers_binding binding) = binding_binds binding.
Proof. destruct binding; reflexivity. Qed.

Fixpoint demote_handlers_binds_statics (scope : list continuation)
    (e : expr) :
  binds_statics (demote_handlers scope e) = binds_statics e
with demote_handlers_handler_binds_statics (scope : list continuation)
    (h : continuation_handler) :
  handler_binds_statics (demote_handlers_handler scope h)
  = handler_binds_statics h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [demote_handlers]; try reflexivity.
    + simpl. apply demote_handlers_binds_statics.
    + simpl. rewrite (demote_handlers_binds_statics scope body). f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite demote_handlers_binding_binds, IH. reflexivity.
    + pose proof (demote_handlers_handler_binds_statics scope h) as Hh.
      destruct (demotable scope k h body); simpl.
      * rewrite strip_traps_binds_statics,
          (demote_handlers_binds_statics (k :: scope) body), <- Hh.
        destruct (demote_handlers_handler scope h). reflexivity.
      * rewrite Hh, (demote_handlers_binds_statics (k :: scope) body).
        reflexivity.
    + simpl.
      rewrite (demote_handlers_binds_statics (map fst handlers ++ scope)
                 body).
      f_equal. generalize (map fst handlers ++ scope). intros inner.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (demote_handlers_handler_binds_statics inner h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply demote_handlers_binds_statics.
Qed.

Definition handler_demoted (h : continuation_handler) : Prop :=
  match h with
  | Handler _ _ handler =>
    ∀ scope : list continuation,
      expr_equivalent handler (demote_handlers scope handler)
  end.

Definition code_demoted (c : code) : Prop :=
  match c with
  | Code _ _ ret exn _ body =>
    expr_equivalent body (demote_handlers [ret; exn] body)
  end.

Definition binding_demoted (binding : static_binding) : Prop :=
  match binding with
  | Bind_code _ c => code_demoted c
  | Bind_deleted_code _ | Bind_symbol _ _ => True
  end.

(* The static bindings of a group, one after the other. *)
Lemma let_static_bindings_demoted (body : expr)
    (todo : list static_binding) :
  Forall binding_demoted todo ->
  ∀ done : list static_binding,
    expr_equivalent (Let_static (done ++ todo) body)
      (Let_static (done ++ map demote_handlers_binding todo) body).
Proof.
  induction 1 as [ | binding todo Hbinding _ IH]; intros done;
    [ apply expr_equivalent_same | ].
  cbn [map].
  assert (Hstep :
    expr_equivalent (Let_static (done ++ binding :: todo) body)
      (Let_static (done ++ demote_handlers_binding binding :: todo) body)).
  { destruct binding as [id c | id | s const];
      try apply expr_equivalent_same.
    destruct c as [params my_closure ret exn metadata code_body].
    exact
      (plug_equivalent
         (Let_static_code done id params my_closure ret exn metadata Hole
            todo body)
         code_body (demote_handlers [ret; exn] code_body)
         (fun _ =>
            eq_sym (demote_handlers_binds_statics [ret; exn] code_body))
         Hbinding). }
  eapply expr_equivalent_trans; [ exact Hstep | ].
  specialize (IH (done ++ [demote_handlers_binding binding])).
  rewrite <- !app_assoc in IH. exact IH.
Qed.

(* The handlers of a recursive [Let_cont], one after the other. *)
Lemma let_cont_recursive_handlers_demoted (inner : list continuation)
    (body : expr) (todo : list (continuation * continuation_handler)) :
  Forall (fun kh => handler_demoted (snd kh)) todo ->
  ∀ done : list (continuation * continuation_handler),
    expr_equivalent (Let_cont_recursive (done ++ todo) body)
      (Let_cont_recursive
         (done
          ++ map (fun kh => (fst kh, demote_handlers_handler inner (snd kh)))
               todo)
         body).
Proof.
  induction 1 as [ | [k h] todo Hh _ IH]; intros done;
    [ apply expr_equivalent_same | ].
  cbn [map fst snd].
  destruct h as [params metadata handler]. cbn [snd handler_demoted] in Hh.
  eapply expr_equivalent_trans.
  - exact
      (plug_equivalent
         (Let_cont_recursive_handler done k params metadata Hole todo body)
         handler (demote_handlers inner handler)
         (fun _ => eq_sym (demote_handlers_binds_statics inner handler))
         (Hh inner)).
  - specialize
      (IH
         (done
          ++ [(k,
               demote_handlers_handler inner
                 (Handler params metadata handler))])).
    rewrite <- !app_assoc in IH. exact IH.
Qed.

Fixpoint demote_handlers_equivalent_aux (e : expr) :
  ∀ scope : list continuation,
    expr_equivalent e (demote_handlers scope e)
with demote_handlers_handler_equivalent (h : continuation_handler) :
  handler_demoted h
with demote_handlers_binding_equivalent (binding : static_binding) :
  binding_demoted binding
with demote_handlers_code_equivalent (c : code) : code_demoted c.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros scope; cbn [demote_handlers]; try apply expr_equivalent_same.
    + exact
        (plug_equivalent (Let_body x kind d Hole) body
           (demote_handlers scope body)
           (fun H => False_ind _ (Bool.diff_false_true H))
           (demote_handlers_equivalent_aux body scope)).
    + eapply expr_equivalent_trans.
      * exact
          (plug_equivalent (Let_static_body bindings Hole) body
             (demote_handlers scope body)
             (fun H => False_ind _ (Bool.diff_false_true H))
             (demote_handlers_equivalent_aux body scope)).
      * apply (let_static_bindings_demoted (demote_handlers scope body)
                 bindings
                 ltac:(induction bindings as [ | binding bindings IH];
                       constructor;
                       [ apply demote_handlers_binding_equivalent
                       | exact IH ])
                 []).
    + pose proof (demote_handlers_handler_equivalent h) as Hh.
      assert (Hinside :
        expr_equivalent (Let_cont_non_recursive k h body)
          (Let_cont_non_recursive k (demote_handlers_handler scope h)
             (demote_handlers (k :: scope) body))).
      { destruct h as [params metadata handler].
        cbn [handler_demoted] in Hh.
        eapply expr_equivalent_trans.
        - exact
            (plug_equivalent
               (Let_cont_non_recursive_body k
                  (Handler params metadata handler) Hole)
               body (demote_handlers (k :: scope) body)
               (fun H => False_ind _ (Bool.diff_false_true H))
               (demote_handlers_equivalent_aux body (k :: scope))).
        - exact
            (plug_equivalent
               (Let_cont_non_recursive_handler k params metadata Hole
                  (demote_handlers (k :: scope) body))
               handler (demote_handlers scope handler)
               (fun H => False_ind _ (Bool.diff_false_true H)) (Hh scope)). }
      destruct (demotable scope k h body); [ | exact Hinside ].
      eapply expr_equivalent_trans; [ exact Hinside | ].
      eapply expr_equivalent_trans.
      * exact
          (plug_equivalent
             (Let_cont_non_recursive_body k (demote_handlers_handler scope h)
                Hole)
             (demote_handlers (k :: scope) body)
             (strip_traps k (demote_handlers (k :: scope) body))
             (fun H => False_ind _ (Bool.diff_false_true H))
             (strip_traps_equivalent k (demote_handlers (k :: scope) body))).
      * apply clear_exn_handler_equivalent.
    + generalize (map fst handlers ++ scope). intros inner.
      eapply expr_equivalent_trans.
      * exact
          (plug_equivalent (Let_cont_recursive_body handlers Hole) body
             (demote_handlers inner body)
             (fun H => False_ind _ (Bool.diff_false_true H))
             (demote_handlers_equivalent_aux body inner)).
      * apply (let_cont_recursive_handlers_demoted inner
                 (demote_handlers inner body) handlers
                 ltac:(induction handlers as [ | [k h] handlers IH];
                       constructor;
                       [ apply demote_handlers_handler_equivalent
                       | exact IH ])
                 []).
  - destruct h as [params metadata handler]. cbn [handler_demoted].
    intros scope. apply demote_handlers_equivalent_aux.
  - destruct binding as [id c | id | s const]; cbn [binding_demoted];
      [ apply demote_handlers_code_equivalent | exact I | exact I ].
  - destruct c as [params my_closure ret exn metadata body].
    cbn [code_demoted]. apply demote_handlers_equivalent_aux.
Qed.

Theorem demote_handlers_equivalent : demote_handlers_equivalent_statement.
Proof.
  intros scope e. exact (demote_handlers_equivalent_aux e scope).
Qed.

(* ======================================================================== *)
(* The binders are kept                                                     *)
(* ======================================================================== *)

Lemma map_fst_handlers (f : continuation_handler -> continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  map fst (map (fun kh => (fst kh, f (snd kh))) handlers) = map fst handlers.
Proof. rewrite map_map. apply map_ext. intros [k h]. reflexivity. Qed.

Lemma is_empty_map {A B : Type} (f : A -> B) (l : list A) :
  Keyed.is_empty (map f l) = Keyed.is_empty l.
Proof. destruct l; reflexivity. Qed.

Fixpoint strip_traps_binders_distinct (k : continuation) (e : expr) :
  binders_distinct (strip_traps k e) = binders_distinct e
with strip_traps_handler_binders_distinct (k : continuation)
    (h : continuation_handler) :
  handler_binders_distinct (strip_traps_handler k h)
  = handler_binders_distinct h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply strip_traps_binders_distinct.
    + rewrite (strip_traps_binders_distinct k body). reflexivity.
    + rewrite (strip_traps_handler_binders_distinct k h),
        (strip_traps_binders_distinct k body).
      reflexivity.
    + rewrite (strip_traps_binders_distinct k body), map_fst_handlers,
        is_empty_map.
      f_equal. f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (strip_traps_handler_binders_distinct k h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite (strip_traps_binders_distinct k handler). reflexivity.
Qed.

Lemma clear_exn_handler_binders_distinct (h : continuation_handler) :
  handler_binders_distinct (clear_exn_handler h) = handler_binders_distinct h.
Proof. destruct h. reflexivity. Qed.

Fixpoint demote_handlers_binders_aux (e : expr) :
  ∀ scope : list continuation,
    binders_distinct (demote_handlers scope e) = binders_distinct e
with demote_handlers_handler_binders (h : continuation_handler) :
  ∀ scope : list continuation,
    handler_binders_distinct (demote_handlers_handler scope h)
    = handler_binders_distinct h
with demote_handlers_binding_binders (binding : static_binding) :
  binding_binders_distinct (demote_handlers_binding binding)
  = binding_binders_distinct binding
with demote_handlers_code_binders (c : code) :
  code_binders_distinct (demote_handlers_code c) = code_binders_distinct c.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros scope; cbn [demote_handlers]; try reflexivity.
    + simpl. apply demote_handlers_binders_aux.
    + simpl. rewrite (demote_handlers_binders_aux body scope). f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite (demote_handlers_binding_binders binding), IH. reflexivity.
    + destruct (demotable scope k h body); simpl.
      * rewrite clear_exn_handler_binders_distinct,
          (demote_handlers_handler_binders h scope),
          strip_traps_binders_distinct,
          (demote_handlers_binders_aux body (k :: scope)).
        reflexivity.
      * rewrite (demote_handlers_handler_binders h scope),
          (demote_handlers_binders_aux body (k :: scope)).
        reflexivity.
    + simpl.
      rewrite (demote_handlers_binders_aux body (map fst handlers ++ scope)),
        map_fst_handlers, is_empty_map.
      f_equal. f_equal.
      generalize (map fst handlers ++ scope). intros inner.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (demote_handlers_handler_binders h inner), IH. reflexivity.
  - destruct h as [params metadata handler]. intros scope. simpl.
    rewrite (demote_handlers_binders_aux handler scope). reflexivity.
  - destruct binding as [id c | id | s const]; simpl; try reflexivity.
    apply demote_handlers_code_binders.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    rewrite (demote_handlers_binders_aux body [ret; exn]). reflexivity.
Qed.

Theorem demote_handlers_binders : demote_handlers_binders_statement.
Proof. intros scope e. exact (demote_handlers_binders_aux e scope). Qed.

(* ======================================================================== *)
(* What an expression does at its leaves                                    *)
(* ======================================================================== *)

(* Whether a call of the expression satisfies [pa], or an [Apply_cont] of
   it [pc], outside its code. The tests that the checks of exception
   handlers make of an expression are all of this form. *)
Section Any_leaf.

Variable pa : apply_expr -> bool.
Variable pc : apply_cont -> bool.

Fixpoint any_leaf (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => any_leaf body
  | Let_cont_non_recursive _ h body => any_leaf_handler h || any_leaf body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => any_leaf_handler h) handlers || any_leaf body
  | Apply ap => pa ap
  | Apply_cont ac => pc ac
  | Switch _ arms => existsb (fun arm => pc (snd arm)) arms
  | Invalid => false
  end
with any_leaf_handler (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => any_leaf handler
  end.

Lemma any_leaf_handler_clear (h : continuation_handler) :
  any_leaf_handler (clear_exn_handler h) = any_leaf_handler h.
Proof. destruct h. reflexivity. Qed.

End Any_leaf.

Lemma existsb_never {A : Type} (l : list A) :
  existsb (fun _ => false) l = false.
Proof. induction l as [ | a l IH]; simpl; [ reflexivity | exact IH ]. Qed.

Fixpoint raises_to_any_leaf (k : continuation) (e : expr) :
  raises_to k e
  = any_leaf (fun ap => Pos.eqb k (ap_exn_continuation ap)) (traps k) e
with handler_raises_to_any_leaf (k : continuation)
    (h : continuation_handler) :
  handler_raises_to k h
  = any_leaf_handler (fun ap => Pos.eqb k (ap_exn_continuation ap)) (traps k)
      h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply raises_to_any_leaf.
    + apply raises_to_any_leaf.
    + rewrite (handler_raises_to_any_leaf k h), (raises_to_any_leaf k body).
      reflexivity.
    + rewrite (raises_to_any_leaf k body). f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_raises_to_any_leaf k h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply raises_to_any_leaf.
Qed.

Fixpoint installs_or_removes_any_leaf (k : continuation) (e : expr) :
  installs_or_removes k e = any_leaf (fun _ => false) (traps_apart k) e
with handler_installs_or_removes_any_leaf (k : continuation)
    (h : continuation_handler) :
  handler_installs_or_removes k h
  = any_leaf_handler (fun _ => false) (traps_apart k) h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply installs_or_removes_any_leaf.
    + apply installs_or_removes_any_leaf.
    + rewrite (handler_installs_or_removes_any_leaf k h),
        (installs_or_removes_any_leaf k body).
      reflexivity.
    + rewrite (installs_or_removes_any_leaf k body). f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_installs_or_removes_any_leaf k h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply installs_or_removes_any_leaf.
Qed.

Fixpoint applies_cont_any_leaf (k : continuation) (e : expr) :
  applies_cont k e
  = any_leaf (fun ap => Pos.eqb k (ap_continuation ap)) (applies k) e
with handler_applies_cont_any_leaf (k : continuation)
    (h : continuation_handler) :
  handler_applies_cont k h
  = any_leaf_handler (fun ap => Pos.eqb k (ap_continuation ap)) (applies k) h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply applies_cont_any_leaf.
    + apply applies_cont_any_leaf.
    + rewrite (handler_applies_cont_any_leaf k h),
        (applies_cont_any_leaf k body).
      reflexivity.
    + rewrite (applies_cont_any_leaf k body). f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_applies_cont_any_leaf k h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply applies_cont_any_leaf.
Qed.

Fixpoint gives_extra_args_any_leaf (k : continuation) (e : expr) :
  gives_extra_args k e
  = any_leaf
      (fun ap =>
         Pos.eqb k (ap_exn_continuation ap) && negb (no_exn_extra_args ap))
      (fun _ => false) e
with handler_gives_extra_args_any_leaf (k : continuation)
    (h : continuation_handler) :
  handler_gives_extra_args k h
  = any_leaf_handler
      (fun ap =>
         Pos.eqb k (ap_exn_continuation ap) && negb (no_exn_extra_args ap))
      (fun _ => false) h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply gives_extra_args_any_leaf.
    + apply gives_extra_args_any_leaf.
    + rewrite (handler_gives_extra_args_any_leaf k h),
        (gives_extra_args_any_leaf k body).
      reflexivity.
    + rewrite (gives_extra_args_any_leaf k body). f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_gives_extra_args_any_leaf k h), IH. reflexivity.
    + symmetry. apply existsb_never.
  - destruct h as [params metadata handler]. simpl.
    apply gives_extra_args_any_leaf.
Qed.

Fixpoint called_with_exn_any_leaf (k : continuation) (e : expr) :
  called_with_exn k e
  = any_leaf (fun ap => Pos.eqb k (ap_exn_continuation ap)) (fun _ => false) e
with handler_called_with_exn_any_leaf (k : continuation)
    (h : continuation_handler) :
  handler_called_with_exn k h
  = any_leaf_handler (fun ap => Pos.eqb k (ap_exn_continuation ap))
      (fun _ => false) h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply called_with_exn_any_leaf.
    + apply called_with_exn_any_leaf.
    + rewrite (handler_called_with_exn_any_leaf k h),
        (called_with_exn_any_leaf k body).
      reflexivity.
    + rewrite (called_with_exn_any_leaf k body). f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_called_with_exn_any_leaf k h), IH. reflexivity.
    + symmetry. apply existsb_never.
  - destruct h as [params metadata handler]. simpl.
    apply called_with_exn_any_leaf.
Qed.

(* Removing the trap actions that name [k] makes no leaf satisfy a test
   that the leaf did not satisfy before. *)
Section Any_leaf_strip.

Variable pa : apply_expr -> bool.
Variables pc pc' : apply_cont -> bool.
Variable k : continuation.
Hypothesis Hleaf :
  ∀ ac : apply_cont, pc' (strip_trap k ac) = true -> pc ac = true.

Fixpoint any_leaf_strip (e : expr) :
  any_leaf pa pc' (strip_traps k e) = true -> any_leaf pa pc e = true
with any_leaf_handler_strip (h : continuation_handler) :
  any_leaf_handler pa pc' (strip_traps_handler k h) = true ->
  any_leaf_handler pa pc h = true.
Proof.
  - destruct e
      as [x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try exact (fun H => H).
    + apply any_leaf_strip.
    + apply any_leaf_strip.
    + rewrite !orb_true_iff.
      intros [H | H];
        [ left; exact (any_leaf_handler_strip h H)
        | right; exact (any_leaf_strip body H) ].
    + rewrite !orb_true_iff.
      intros [H | H]; [ left | right; exact (any_leaf_strip body H) ].
      revert H.
      induction handlers as [ | [k' h] handlers IH]; simpl;
        [ exact (fun H => H) | ].
      rewrite !orb_true_iff.
      intros [H | H];
        [ left; exact (any_leaf_handler_strip h H) | right; exact (IH H) ].
    + apply Hleaf.
    + induction arms as [ | [i ac] arms IH]; simpl; [ exact (fun H => H) | ].
      rewrite !orb_true_iff.
      intros [H | H]; [ left; exact (Hleaf ac H) | right; exact (IH H) ].
  - destruct h as [params metadata handler]. simpl. apply any_leaf_strip.
Qed.

End Any_leaf_strip.

(* What the tests say of an [Apply_cont] without its trap action. *)
Lemma traps_strip_trap (k k' : continuation) (ac : apply_cont) :
  traps k' (strip_trap k ac) = true -> traps k' ac = true.
Proof.
  unfold strip_trap. destruct (traps k ac); [ discriminate | ].
  exact (fun H => H).
Qed.

Lemma traps_strip_trap_same (k : continuation) (ac : apply_cont) :
  traps k (strip_trap k ac) = false.
Proof.
  unfold strip_trap. destruct (traps k ac) eqn:Htraps;
    [ reflexivity | exact Htraps ].
Qed.

Lemma traps_apart_strip_trap (k k' : continuation) (ac : apply_cont) :
  traps_apart k' (strip_trap k ac) = true -> traps_apart k' ac = true.
Proof.
  unfold strip_trap. destruct (traps k ac); [ discriminate | ].
  exact (fun H => H).
Qed.

Lemma applies_strip_trap (k k' : continuation) (ac : apply_cont) :
  k' <> k -> applies k' (strip_trap k ac) = true -> applies k' ac = true.
Proof.
  intros Hne. unfold strip_trap.
  destruct (traps k ac) eqn:Htraps; [ | exact (fun H => H) ].
  unfold applies, is_raise, traps in *.
  destruct ac as [c args [[handler | handler kind] | ]]; simpl in *;
    try discriminate; try exact (fun H => H).
  apply Pos.eqb_eq in Htraps. subst handler.
  rewrite !andb_true_iff. intros [Hc _]. split; [ exact Hc | ].
  apply Pos.eqb_eq in Hc. subst c. apply negb_true_iff.
  apply Pos.eqb_neq. exact Hne.
Qed.

Lemma raises_to_strip (k k' : continuation) (e : expr) :
  raises_to k' (strip_traps k e) = true -> raises_to k' e = true.
Proof.
  rewrite !raises_to_any_leaf. apply any_leaf_strip.
  intros ac. apply traps_strip_trap.
Qed.

Lemma raises_to_strip_same (k : continuation) (e : expr) :
  raises_to k (strip_traps k e) = true -> called_with_exn k e = true.
Proof.
  rewrite raises_to_any_leaf, called_with_exn_any_leaf. apply any_leaf_strip.
  intros ac. rewrite traps_strip_trap_same. discriminate.
Qed.

Lemma applies_cont_strip (k k' : continuation) (e : expr) :
  k' <> k ->
  applies_cont k' (strip_traps k e) = true -> applies_cont k' e = true.
Proof.
  intros Hne. rewrite !applies_cont_any_leaf. apply any_leaf_strip.
  intros ac. apply applies_strip_trap. exact Hne.
Qed.

Lemma called_with_exn_strip (k k' : continuation) (e : expr) :
  called_with_exn k' (strip_traps k e) = true -> called_with_exn k' e = true.
Proof.
  rewrite !called_with_exn_any_leaf. apply any_leaf_strip.
  intros ac. exact (fun H => H).
Qed.

(* What [demotable] says. *)
Lemma demotable_true (scope : list continuation) (k : continuation)
    (h : continuation_handler) (body : expr) :
  demotable scope k h body = true ->
  handler_is_exn_handler h = true
  ∧ called_with_exn k body = false
  ∧ ~ In k scope
  ∧ binds_continuation k body = false.
Proof.
  unfold demotable. rewrite !andb_true_iff, !negb_true_iff.
  intros [[[Hmarked Hcalled] Hscope] Hbinds].
  split; [ exact Hmarked | ]. split; [ exact Hcalled | ].
  split; [ apply among_false; exact Hscope | exact Hbinds ].
Qed.

(* The rewrite makes no leaf satisfy a test that the leaf did not satisfy
   before, if removing the trap actions that name a continuation outside
   [scope0] does not; [scope0] are names in scope, which the rewrite
   does not demote. *)
Section Any_leaf_demote.

Variable pa : apply_expr -> bool.
Variable pc : apply_cont -> bool.
Variable scope0 : list continuation.
Hypothesis Hleaf :
  ∀ (k : continuation) (ac : apply_cont),
    ~ In k scope0 -> pc (strip_trap k ac) = true -> pc ac = true.

Fixpoint any_leaf_demote (e : expr) :
  ∀ scope : list continuation,
    incl scope0 scope ->
    any_leaf pa pc (demote_handlers scope e) = true -> any_leaf pa pc e = true
with any_leaf_handler_demote (h : continuation_handler) :
  ∀ scope : list continuation,
    incl scope0 scope ->
    any_leaf_handler pa pc (demote_handlers_handler scope h) = true ->
    any_leaf_handler pa pc h = true.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros scope Hscope; cbn [demote_handlers]; try exact (fun H => H).
    + simpl. apply any_leaf_demote, Hscope.
    + simpl. apply any_leaf_demote, Hscope.
    + assert (Hbody :
        any_leaf pa pc (demote_handlers (k :: scope) body) = true ->
        any_leaf pa pc body = true).
      { apply any_leaf_demote. apply incl_tl, Hscope. }
      destruct (demotable scope k h body) eqn:Hdemotable; simpl.
      * apply demotable_true in Hdemotable as (_ & _ & Hk & _).
        rewrite any_leaf_handler_clear, !orb_true_iff.
        intros [H | H];
          [ left; exact (any_leaf_handler_demote h scope Hscope H) | ].
        right. apply Hbody.
        apply (any_leaf_strip pa pc pc k); [ | exact H ].
        intros ac. apply Hleaf. intros Hin. exact (Hk (Hscope k Hin)).
      * rewrite !orb_true_iff.
        intros [H | H];
          [ left; exact (any_leaf_handler_demote h scope Hscope H)
          | right; exact (Hbody H) ].
    + simpl.
      assert (Hinner : incl scope0 (map fst handlers ++ scope))
        by (apply incl_appr, Hscope).
      revert Hinner. generalize (map fst handlers ++ scope).
      intros inner Hinner. rewrite !orb_true_iff.
      intros [H | H];
        [ left | right; exact (any_leaf_demote body inner Hinner H) ].
      revert H.
      induction handlers as [ | [k h] handlers IH]; simpl;
        [ exact (fun H => H) | ].
      rewrite !orb_true_iff.
      intros [H | H];
        [ left; exact (any_leaf_handler_demote h inner Hinner H)
        | right; exact (IH H) ].
  - destruct h as [params metadata handler]. intros scope Hscope. simpl.
    apply any_leaf_demote, Hscope.
Qed.

End Any_leaf_demote.

Lemma raises_to_demote (scope : list continuation) (k : continuation)
    (e : expr) :
  raises_to k (demote_handlers scope e) = true -> raises_to k e = true.
Proof.
  rewrite !raises_to_any_leaf.
  apply (any_leaf_demote _ _ []); [ | apply incl_nil_l ].
  intros k0 ac _. apply traps_strip_trap.
Qed.

Lemma installs_or_removes_demote (scope : list continuation)
    (k : continuation) (e : expr) :
  installs_or_removes k (demote_handlers scope e) = true ->
  installs_or_removes k e = true.
Proof.
  rewrite !installs_or_removes_any_leaf.
  apply (any_leaf_demote _ _ []); [ | apply incl_nil_l ].
  intros k0 ac _. apply traps_apart_strip_trap.
Qed.

Lemma gives_extra_args_demote (scope : list continuation) (k : continuation)
    (e : expr) :
  gives_extra_args k (demote_handlers scope e) = true ->
  gives_extra_args k e = true.
Proof.
  rewrite !gives_extra_args_any_leaf.
  apply (any_leaf_demote _ _ []); [ | apply incl_nil_l ].
  intros k0 ac _. exact (fun H => H).
Qed.

Lemma called_with_exn_demote (scope : list continuation) (k : continuation)
    (e : expr) :
  called_with_exn k (demote_handlers scope e) = true ->
  called_with_exn k e = true.
Proof.
  rewrite !called_with_exn_any_leaf.
  apply (any_leaf_demote _ _ []); [ | apply incl_nil_l ].
  intros k0 ac _. exact (fun H => H).
Qed.

Lemma applies_cont_demote (scope : list continuation) (k : continuation)
    (e : expr) :
  In k scope ->
  applies_cont k (demote_handlers scope e) = true -> applies_cont k e = true.
Proof.
  intros Hk. rewrite !applies_cont_any_leaf.
  apply (any_leaf_demote _ _ [k]).
  - intros k0 ac Hk0. apply applies_strip_trap.
    intros ->. apply Hk0. left. reflexivity.
  - intros k0 [<- | []]. exact Hk.
Qed.

(* The continuations bound are the same. *)
Fixpoint binds_continuation_strip (k k' : continuation) (e : expr) :
  binds_continuation k' (strip_traps k e) = binds_continuation k' e
with handler_binds_continuation_strip (k k' : continuation)
    (h : continuation_handler) :
  handler_binds_continuation k' (strip_traps_handler k h)
  = handler_binds_continuation k' h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k1 h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply binds_continuation_strip.
    + apply binds_continuation_strip.
    + rewrite (handler_binds_continuation_strip k k' h),
        (binds_continuation_strip k k' body).
      reflexivity.
    + rewrite (binds_continuation_strip k k' body). f_equal.
      induction handlers as [ | [k1 h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_binds_continuation_strip k k' h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply binds_continuation_strip.
Qed.

Lemma handler_binds_continuation_clear (k : continuation)
    (h : continuation_handler) :
  handler_binds_continuation k (clear_exn_handler h)
  = handler_binds_continuation k h.
Proof. destruct h. reflexivity. Qed.

Fixpoint binds_continuation_demote (k' : continuation) (e : expr) :
  ∀ scope : list continuation,
    binds_continuation k' (demote_handlers scope e) = binds_continuation k' e
with handler_binds_continuation_demote (k' : continuation)
    (h : continuation_handler) :
  ∀ scope : list continuation,
    handler_binds_continuation k' (demote_handlers_handler scope h)
    = handler_binds_continuation k' h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros scope; cbn [demote_handlers]; try reflexivity.
    + simpl. apply binds_continuation_demote.
    + simpl. apply binds_continuation_demote.
    + destruct (demotable scope k h body); simpl.
      * rewrite handler_binds_continuation_clear,
          (handler_binds_continuation_demote k' h scope),
          binds_continuation_strip,
          (binds_continuation_demote k' body (k :: scope)).
        reflexivity.
      * rewrite (handler_binds_continuation_demote k' h scope),
          (binds_continuation_demote k' body (k :: scope)).
        reflexivity.
    + simpl.
      rewrite (binds_continuation_demote k' body (map fst handlers ++ scope)).
      f_equal. generalize (map fst handlers ++ scope). intros inner.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_binds_continuation_demote k' h inner), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros scope. simpl.
    apply binds_continuation_demote.
Qed.

(* The continuations that an expression does not bind are the same. *)
Lemma without_app (bound l l' : list positive) :
  without bound (l ++ l') = without bound l ++ without bound l'.
Proof. apply filter_app. Qed.

Lemma without_without (bound bound' l : list positive) :
  without bound (without bound' l) = without bound' (without bound l).
Proof.
  unfold without. induction l as [ | name l IH]; simpl; [ reflexivity | ].
  destruct (among name bound) eqn:Hb, (among name bound') eqn:Hb'; simpl;
    rewrite ?Hb, ?Hb'; simpl; rewrite IH; reflexivity.
Qed.

Lemma free_continuations_strip_trap (k : continuation) (ac : apply_cont) :
  without [k] (apply_cont_continuations (strip_trap k ac))
  = without [k] (apply_cont_continuations ac).
Proof.
  unfold strip_trap. destruct (traps k ac) eqn:Htraps; [ | reflexivity ].
  unfold traps in Htraps.
  destruct ac as [c args [action | ]]; simpl in *; [ | discriminate ].
  apply Pos.eqb_eq in Htraps. rewrite <- Htraps.
  unfold without, among. simpl. rewrite Pos.eqb_refl. simpl.
  destruct (Pos.eqb c k); reflexivity.
Qed.

Fixpoint free_continuations_strip (k : continuation) (e : expr) :
  without [k] (free_continuations (strip_traps k e))
  = without [k] (free_continuations e)
with handler_free_continuations_strip (k : continuation)
    (h : continuation_handler) :
  without [k] (handler_free_continuations (strip_traps_handler k h))
  = without [k] (handler_free_continuations h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k1 h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply free_continuations_strip.
    + apply free_continuations_strip.
    + rewrite !without_app, (handler_free_continuations_strip k h),
        !(without_without [k] [k1]), (free_continuations_strip k body).
      reflexivity.
    + rewrite map_fst_handlers, !(without_without [k] (map fst handlers)),
        !without_app, (free_continuations_strip k body).
      f_equal. f_equal.
      induction handlers as [ | [k1 h] handlers IH]; simpl; [ reflexivity | ].
      rewrite !without_app, (handler_free_continuations_strip k h), IH.
      reflexivity.
    + apply free_continuations_strip_trap.
    + induction arms as [ | [i ac] arms IH]; [ reflexivity | ].
      cbn [flat_map map fst snd].
      rewrite !without_app, free_continuations_strip_trap, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply free_continuations_strip.
Qed.

Lemma handler_free_continuations_clear (h : continuation_handler) :
  handler_free_continuations (clear_exn_handler h)
  = handler_free_continuations h.
Proof. destruct h. reflexivity. Qed.

Fixpoint free_continuations_demote (e : expr) :
  ∀ scope : list continuation,
    free_continuations (demote_handlers scope e) = free_continuations e
with handler_free_continuations_demote (h : continuation_handler) :
  ∀ scope : list continuation,
    handler_free_continuations (demote_handlers_handler scope h)
    = handler_free_continuations h.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros scope; cbn [demote_handlers]; try reflexivity.
    + simpl. apply free_continuations_demote.
    + simpl. apply free_continuations_demote.
    + destruct (demotable scope k h body); simpl.
      * rewrite handler_free_continuations_clear,
          (handler_free_continuations_demote h scope),
          free_continuations_strip,
          (free_continuations_demote body (k :: scope)).
        reflexivity.
      * rewrite (handler_free_continuations_demote h scope),
          (free_continuations_demote body (k :: scope)).
        reflexivity.
    + simpl.
      rewrite map_fst_handlers,
        (free_continuations_demote body (map fst handlers ++ scope)).
      f_equal. f_equal.
      generalize (map fst handlers ++ scope). intros inner.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_free_continuations_demote h inner), IH. reflexivity.
  - destruct h as [params metadata handler]. intros scope. simpl.
    apply free_continuations_demote.
Qed.

(* ======================================================================== *)
(* Exception handlers stay apart from the other continuations               *)
(* ======================================================================== *)

Lemma negb_mono (b b' : bool) :
  (b' = true -> b = true) -> negb b = true -> negb b' = true.
Proof. destruct b, b'; simpl; auto. Qed.

Lemma ways_out_demote (exn : continuation) (scope : list continuation)
    (e : expr) :
  ways_out_well_formed exn e = true ->
  ways_out_well_formed exn (demote_handlers scope e) = true.
Proof.
  unfold ways_out_well_formed. rewrite free_continuations_demote.
  rewrite !forallb_forall. intros Hall k Hk. specialize (Hall k Hk).
  apply andb_true_iff in Hall as [Huse Hextra].
  apply andb_true_iff. split.
  - destruct (Pos.eqb k exn).
    + revert Huse. apply negb_mono, installs_or_removes_demote.
    + revert Huse. apply negb_mono, raises_to_demote.
  - revert Hextra. apply negb_mono, gives_extra_args_demote.
Qed.

Lemma strip_traps_handler_is_exn_handler (k : continuation)
    (h : continuation_handler) :
  handler_is_exn_handler (strip_traps_handler k h) = handler_is_exn_handler h.
Proof. destruct h. reflexivity. Qed.

Lemma strip_traps_handler_takes_exception (k : continuation)
    (h : continuation_handler) :
  takes_exception (strip_traps_handler k h) = takes_exception h.
Proof. destruct h as [[ | x params] metadata handler]; reflexivity. Qed.

Lemma demote_handlers_handler_is_exn_handler (scope : list continuation)
    (h : continuation_handler) :
  handler_is_exn_handler (demote_handlers_handler scope h)
  = handler_is_exn_handler h.
Proof. destruct h. reflexivity. Qed.

Lemma demote_handlers_handler_takes_exception (scope : list continuation)
    (h : continuation_handler) :
  takes_exception (demote_handlers_handler scope h) = takes_exception h.
Proof. destruct h as [[ | x params] metadata handler]; reflexivity. Qed.

Lemma clear_exn_handler_is_exn_handler (h : continuation_handler) :
  handler_is_exn_handler (clear_exn_handler h) = false.
Proof. destruct h. reflexivity. Qed.

Lemma handler_exn_handlers_well_formed_clear (h : continuation_handler) :
  handler_exn_handlers_well_formed (clear_exn_handler h)
  = handler_exn_handlers_well_formed h.
Proof. destruct h. reflexivity. Qed.

(* A handler of a recursive group that raises to [k] after the trap
   actions are removed, or after the rewrite, raised to it before. *)
Lemma handlers_raise_to_strip (k k' : continuation)
    (handlers : list (continuation * continuation_handler)) :
  existsb (fun '(_, h) => handler_raises_to k' h)
    (map (fun kh => (fst kh, strip_traps_handler k (snd kh))) handlers)
  = true ->
  existsb (fun '(_, h) => handler_raises_to k' h) handlers = true.
Proof.
  induction handlers as [ | [k1 [params metadata handler]] handlers IH];
    simpl; [ exact (fun H => H) | ].
  rewrite !orb_true_iff.
  intros [H | H]; [ left; exact (raises_to_strip k k' handler H) | ].
  right. exact (IH H).
Qed.

Lemma handlers_raise_to_demote (inner : list continuation) (k' : continuation)
    (handlers : list (continuation * continuation_handler)) :
  existsb (fun '(_, h) => handler_raises_to k' h)
    (map (fun kh => (fst kh, demote_handlers_handler inner (snd kh)))
       handlers)
  = true ->
  existsb (fun '(_, h) => handler_raises_to k' h) handlers = true.
Proof.
  induction handlers as [ | [k1 [params metadata handler]] handlers IH];
    simpl; [ exact (fun H => H) | ].
  rewrite !orb_true_iff.
  intros [H | H]; [ left; exact (raises_to_demote inner k' handler H) | ].
  right. exact (IH H).
Qed.

(* The handlers of a recursive group, under a rewrite of the handlers
   and of the body that adds no raise and keeps the marks. *)
Lemma group_exn_handlers_rewritten
    (f : continuation_handler -> continuation_handler) (body body' : expr)
    (group todo : list (continuation * continuation_handler)) :
  (∀ k0 : continuation,
    raises_to k0 body' = true -> raises_to k0 body = true) ->
  (∀ k0 : continuation,
    existsb (fun '(_, h') => handler_raises_to k0 h')
      (map (fun kh => (fst kh, f (snd kh))) group)
    = true ->
    existsb (fun '(_, h') => handler_raises_to k0 h') group = true) ->
  (∀ h : continuation_handler,
    handler_is_exn_handler (f h) = handler_is_exn_handler h) ->
  Forall
    (fun kh =>
       handler_exn_handlers_well_formed (snd kh) = true ->
       handler_exn_handlers_well_formed (f (snd kh)) = true)
    todo ->
  forallb
    (fun '(k0, h) =>
       (negb (handler_is_exn_handler h)
        && negb (raises_to k0 body)
        && negb (existsb (fun '(_, h') => handler_raises_to k0 h') group)
        && handler_exn_handlers_well_formed h)%bool)
    todo
  = true ->
  forallb
    (fun '(k0, h) =>
       (negb (handler_is_exn_handler h)
        && negb (raises_to k0 body')
        && negb
             (existsb (fun '(_, h') => handler_raises_to k0 h')
                (map (fun kh => (fst kh, f (snd kh))) group))
        && handler_exn_handlers_well_formed h)%bool)
    (map (fun kh => (fst kh, f (snd kh))) todo)
  = true.
Proof.
  intros Hbody Hgroup Hmark.
  induction 1 as [ | [k1 h] todo Hh _ IH]; simpl; [ reflexivity | ].
  rewrite !andb_true_iff.
  intros [[[[Hmarked Hbody1] Hgroup1] Hwf] Hrest].
  split; [ | exact (IH Hrest) ].
  split; [ split; [ split | ] | ].
  - rewrite Hmark. exact Hmarked.
  - revert Hbody1. apply negb_mono, Hbody.
  - revert Hgroup1. apply negb_mono, Hgroup.
  - exact (Hh Hwf).
Qed.

Fixpoint exn_handlers_strip (k : continuation) (e : expr) :
  binds_continuation k e = false ->
  exn_handlers_well_formed e = true ->
  exn_handlers_well_formed (strip_traps k e) = true
with handler_exn_handlers_strip (k : continuation)
    (h : continuation_handler) :
  handler_binds_continuation k h = false ->
  handler_exn_handlers_well_formed h = true ->
  handler_exn_handlers_well_formed (strip_traps_handler k h) = true.
Proof.
  - destruct e
      as [x kind d body | bindings body | k1 h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try (intros _; exact (fun H => H)).
    + apply exn_handlers_strip.
    + intros Hbinds. rewrite !andb_true_iff. intros [Hbindings Hbody].
      split;
        [ exact Hbindings | exact (exn_handlers_strip k body Hbinds Hbody) ].
    + rewrite !orb_false_iff. intros [[Hk1 Hbinds_h] Hbinds_body].
      apply Pos.eqb_neq in Hk1.
      rewrite !andb_true_iff. intros [[Huse Hh] Hbody].
      split; [ split | exact (exn_handlers_strip k body Hbinds_body Hbody) ];
        [ | exact (handler_exn_handlers_strip k h Hbinds_h Hh) ].
      rewrite strip_traps_handler_is_exn_handler,
        strip_traps_handler_takes_exception.
      destruct (handler_is_exn_handler h).
      * apply andb_true_iff in Huse as [Happlies Htakes].
        apply andb_true_iff. split; [ | exact Htakes ].
        revert Happlies. apply negb_mono, applies_cont_strip.
        intros ->. apply Hk1. reflexivity.
      * revert Huse. apply negb_mono, raises_to_strip.
    + rewrite !orb_false_iff. intros [Hbinds_handlers Hbinds_body].
      rewrite !andb_true_iff. intros [Hhandlers Hbody].
      split; [ | exact (exn_handlers_strip k body Hbinds_body Hbody) ].
      assert (Hrec :
        Forall
          (fun kh =>
             handler_exn_handlers_well_formed (snd kh) = true ->
             handler_exn_handlers_well_formed
               (strip_traps_handler k (snd kh))
             = true)
          handlers).
      { clear Hhandlers. revert Hbinds_handlers.
        induction handlers as [ | [k1 h] handlers IH]; simpl; intros Hbinds;
          constructor.
        - apply orb_false_iff in Hbinds as [Hhead _].
          apply orb_false_iff in Hhead as [_ Hbinds_h].
          exact (handler_exn_handlers_strip k h Hbinds_h).
        - apply orb_false_iff in Hbinds as [_ Hrest]. exact (IH Hrest). }
      exact
        (group_exn_handlers_rewritten (strip_traps_handler k) body
           (strip_traps k body) handlers handlers
           (fun k0 => raises_to_strip k k0 body)
           (fun k0 => handlers_raise_to_strip k k0 handlers)
           (strip_traps_handler_is_exn_handler k) Hrec Hhandlers).
  - destruct h as [params metadata handler]. simpl.
    apply exn_handlers_strip.
Qed.

Fixpoint exn_handlers_demote (e : expr) :
  ∀ scope : list continuation,
    exn_handlers_well_formed e = true ->
    exn_handlers_well_formed (demote_handlers scope e) = true
with handler_exn_handlers_demote (h : continuation_handler) :
  ∀ scope : list continuation,
    handler_exn_handlers_well_formed h = true ->
    handler_exn_handlers_well_formed (demote_handlers_handler scope h) = true
with binding_exn_handlers_demote (binding : static_binding) :
  binding_exn_handlers_well_formed binding = true ->
  binding_exn_handlers_well_formed (demote_handlers_binding binding) = true
with code_exn_handlers_demote (c : code) :
  code_exn_handlers_well_formed c = true ->
  code_exn_handlers_well_formed (demote_handlers_code c) = true.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros scope; cbn [demote_handlers]; try exact (fun H => H).
    + simpl. apply exn_handlers_demote.
    + simpl. rewrite !andb_true_iff. intros [Hbindings Hbody].
      split; [ | exact (exn_handlers_demote body scope Hbody) ].
      revert Hbindings.
      induction bindings as [ | binding bindings IH]; simpl;
        [ exact (fun H => H) | ].
      rewrite !andb_true_iff. intros [Hbinding Hrest].
      split; [ exact (binding_exn_handlers_demote binding Hbinding) | ].
      exact (IH Hrest).
    + simpl. rewrite !andb_true_iff. intros [[Huse Hh] Hbody].
      pose proof (handler_exn_handlers_demote h scope Hh) as Hh'.
      pose proof (exn_handlers_demote body (k :: scope) Hbody) as Hbody'.
      destruct (demotable scope k h body) eqn:Hdemotable; simpl.
      * apply demotable_true in Hdemotable as (_ & Hcalled & _ & Hbinds).
        rewrite clear_exn_handler_is_exn_handler,
          handler_exn_handlers_well_formed_clear, !andb_true_iff.
        split; [ split; [ | exact Hh' ] | ].
        -- apply negb_true_iff.
           destruct
             (raises_to k (strip_traps k (demote_handlers (k :: scope) body)))
             eqn:Hraises; [ | reflexivity ].
           apply raises_to_strip_same, called_with_exn_demote in Hraises.
           congruence.
        -- apply exn_handlers_strip; [ | exact Hbody' ].
           rewrite binds_continuation_demote. exact Hbinds.
      * rewrite demote_handlers_handler_is_exn_handler,
          demote_handlers_handler_takes_exception, !andb_true_iff.
        split; [ split; [ | exact Hh' ] | exact Hbody' ].
        destruct (handler_is_exn_handler h).
        -- apply andb_true_iff in Huse as [Happlies Htakes].
           apply andb_true_iff. split; [ | exact Htakes ].
           revert Happlies. apply negb_mono, applies_cont_demote.
           left. reflexivity.
        -- revert Huse. apply negb_mono, raises_to_demote.
    + simpl. rewrite !andb_true_iff. intros [Hhandlers Hbody].
      split;
        [ | exact (exn_handlers_demote body (map fst handlers ++ scope)
                     Hbody) ].
      revert Hhandlers. generalize (map fst handlers ++ scope).
      intros inner Hhandlers.
      assert (Hrec :
        Forall
          (fun kh =>
             handler_exn_handlers_well_formed (snd kh) = true ->
             handler_exn_handlers_well_formed
               (demote_handlers_handler inner (snd kh))
             = true)
          handlers).
      { clear Hhandlers Hbody.
        induction handlers as [ | [k1 h] handlers IH]; constructor;
          [ exact (handler_exn_handlers_demote h inner) | exact IH ]. }
      exact
        (group_exn_handlers_rewritten (demote_handlers_handler inner) body
           (demote_handlers inner body) handlers handlers
           (fun k0 => raises_to_demote inner k0 body)
           (fun k0 => handlers_raise_to_demote inner k0 handlers)
           (demote_handlers_handler_is_exn_handler inner) Hrec Hhandlers).
  - destruct h as [params metadata handler]. intros scope. simpl.
    apply exn_handlers_demote.
  - destruct binding as [id c | id | s const]; simpl;
      try exact (fun H => H).
    apply code_exn_handlers_demote.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    rewrite !andb_true_iff. intros [[Hways Happlies] Hbody].
    split; [ split | exact (exn_handlers_demote body [ret; exn] Hbody) ].
    + exact (ways_out_demote exn [ret; exn] body Hways).
    + revert Happlies. apply negb_mono, applies_cont_demote.
      right. left. reflexivity.
Qed.

Theorem demote_handlers_keeps_exn_handlers :
  demote_handlers_keeps_exn_handlers_statement.
Proof.
  intros unit_exn scope e Hscope. unfold unit_exn_handlers_well_formed.
  rewrite !andb_true_iff. intros [[Hways Happlies] He].
  split; [ split | exact (exn_handlers_demote e scope He) ].
  - exact (ways_out_demote unit_exn scope e Hways).
  - revert Happlies. apply negb_mono, applies_cont_demote. exact Hscope.
Qed.

(* ======================================================================== *)
(* Demands, one below another                                               *)
(* ======================================================================== *)

Lemma entry_of_none (k : continuation) (ds : demands) :
  (∀ d : continuation * trap_stack, In d ds -> fst d <> k) ->
  entry_of k ds = Not_entered.
Proof.
  intros Hnone. unfold entry_of.
  destruct (filter (fun d => Pos.eqb (fst d) k) ds) as [ | d others]
    eqn:Hfilter; [ reflexivity | ].
  assert (Hin : In d (filter (fun d => Pos.eqb (fst d) k) ds))
    by (rewrite Hfilter; left; reflexivity).
  apply filter_In in Hin as [Hin Hk]. apply Pos.eqb_eq in Hk.
  contradiction (Hnone d Hin Hk).
Qed.

Lemma entry_of_one (k : continuation) (ds : demands) (ts : trap_stack) :
  (∀ d : continuation * trap_stack, In d ds -> fst d = k -> snd d = ts) ->
  entry_of k ds = Not_entered ∨ entry_of k ds = Entered_with ts.
Proof.
  intros Hone. unfold entry_of.
  assert (Hall :
    ∀ t : trap_stack,
      In t (map snd (filter (fun d => Pos.eqb (fst d) k) ds)) -> t = ts).
  { intros t Hin. apply in_map_iff in Hin as (d & <- & Hin).
    apply filter_In in Hin as [Hin Hk]. apply Pos.eqb_eq in Hk.
    exact (Hone d Hin Hk). }
  destruct (map snd (filter (fun d => Pos.eqb (fst d) k) ds))
    as [ | t others]; [ left; reflexivity | right ].
  rewrite (Hall t (or_introl eq_refl)).
  assert (Hothers : forallb (trap_stack_eqb ts) others = true).
  { apply forallb_forall. intros t' Hin. apply trap_stack_eqb_eq.
    symmetry. apply Hall. right. exact Hin. }
  rewrite Hothers. reflexivity.
Qed.

(* The demands that leave a scope have none of its names on their
   stacks. *)
Lemma leave_scope_stacks (ks : list continuation) (ds outer : demands) :
  leave_scope ks ds = Some outer ->
  ∀ d : continuation * trap_stack,
    In d outer -> ∀ k : continuation, In k ks -> ~ In k (snd d).
Proof.
  unfold leave_scope. intros Hleave d Hin k Hk.
  set (outer0 := filter (fun d => negb (among (fst d) ks)) ds) in *.
  destruct
    (forallb (fun d => forallb (fun k => negb (among k (snd d))) ks) outer0)
    eqn:Hall; [ | discriminate ].
  injection Hleave as <-.
  rewrite forallb_forall in Hall. specialize (Hall d Hin).
  rewrite forallb_forall in Hall. apply among_false, negb_true_iff, Hall, Hk.
Qed.

(* [ds'] is below [ds] under a map [m] of stacks: each demand of [ds'] is
   one of [ds] with its stack mapped. With the identity, it is that
   [ds'] is among [ds]. *)
Section Sub.

Variable m : trap_stack -> trap_stack.
Hypothesis m_shrinks :
  ∀ (k : continuation) (ts : trap_stack), In k (m ts) -> In k ts.

Definition sub (ds ds' : demands) : Prop :=
  ∀ d' : continuation * trap_stack,
    In d' ds' ->
    ∃ d : continuation * trap_stack,
      In d ds ∧ fst d' = fst d ∧ snd d' = m (snd d).

Lemma sub_nil (ds : demands) : sub ds [].
Proof. intros d' []. Qed.

Lemma sub_app (ds1 ds1' ds2 ds2' : demands) :
  sub ds1 ds1' -> sub ds2 ds2' -> sub (ds1 ++ ds2) (ds1' ++ ds2').
Proof.
  intros H1 H2 d' Hin. apply in_app_or in Hin as [Hin | Hin].
  - destruct (H1 d' Hin) as (d & Hd & Heq). exists d.
    split; [ apply in_or_app; left; exact Hd | exact Heq ].
  - destruct (H2 d' Hin) as (d & Hd & Heq). exists d.
    split; [ apply in_or_app; right; exact Hd | exact Heq ].
Qed.

Lemma sub_weaken (ds ds_more ds' : demands) :
  incl ds ds_more -> sub ds ds' -> sub ds_more ds'.
Proof.
  intros Hincl Hsub d' Hin. destruct (Hsub d' Hin) as (d & Hd & Heq).
  exists d. split; [ exact (Hincl d Hd) | exact Heq ].
Qed.

Lemma among_shrinks (k : continuation) (ts : trap_stack) :
  among k ts = false -> among k (m ts) = false.
Proof.
  rewrite !among_false. intros Hk Hin. exact (Hk (m_shrinks k ts Hin)).
Qed.

Lemma sub_entry (k : continuation) (ds ds' : demands) :
  sub ds ds' ->
  match entry_of k ds with
  | Not_entered => entry_of k ds' = Not_entered
  | Entered_with ts =>
    entry_of k ds' = Not_entered ∨ entry_of k ds' = Entered_with (m ts)
  | Entered_with_two_stacks => True
  end.
Proof.
  intros Hsub. destruct (entry_of k ds) as [ | ts | ] eqn:Hentry;
    [ | | exact I ].
  - apply entry_of_none. intros d' Hin Hk.
    destruct (Hsub d' Hin) as (d & Hd & Hfst & _).
    apply (entry_of_not_entered k ds Hentry d Hd). congruence.
  - apply entry_of_one. intros d' Hin Hk.
    destruct (Hsub d' Hin) as (d & Hd & Hfst & ->).
    rewrite (entry_of_entered k ds ts Hentry d Hd); [ reflexivity | ].
    congruence.
Qed.

Lemma sub_leave (ks : list continuation) (ds ds' outer : demands) :
  sub ds ds' ->
  leave_scope ks ds = Some outer ->
  ∃ outer' : demands, leave_scope ks ds' = Some outer' ∧ sub outer outer'.
Proof.
  intros Hsub Hleave.
  assert (Hfrom :
    ∀ d' : continuation * trap_stack,
      In d' (filter (fun d => negb (among (fst d) ks)) ds') ->
      ∃ d : continuation * trap_stack,
        In d outer ∧ fst d' = fst d ∧ snd d' = m (snd d)
        ∧ ∀ k : continuation, In k ks -> ~ In k (snd d)).
  { intros d' Hin. apply filter_In in Hin as [Hin Hk].
    apply negb_true_iff in Hk.
    destruct (Hsub d' Hin) as (d & Hd & Hfst & Hsnd).
    assert (Hk' : among (fst d) ks = false) by (rewrite <- Hfst; exact Hk).
    destruct (leave_scope_in ks ds outer Hleave d Hd Hk') as [Houter Hstack].
    exists d. auto. }
  assert (Hall :
    forallb (fun d => forallb (fun k => negb (among k (snd d))) ks)
      (filter (fun d => negb (among (fst d) ks)) ds')
    = true).
  { apply forallb_forall. intros d' Hin.
    destruct (Hfrom d' Hin) as (d & _ & _ & Hsnd & Hstack).
    apply forallb_forall. intros k Hk. apply negb_true_iff, among_false.
    intros Hin'. apply (Hstack k Hk), (m_shrinks k).
    rewrite <- Hsnd. exact Hin'. }
  exists (filter (fun d => negb (among (fst d) ks)) ds'). split.
  - unfold leave_scope. rewrite Hall. reflexivity.
  - intros d' Hin. destruct (Hfrom d' Hin) as (d & Hd & Hfst & Hsnd & _).
    exists d. auto.
Qed.

(* A non-recursive [Let_cont] whose body and whose handler have demands
   below those they had. *)
Lemma sub_let_cont (k : continuation) (h h' : continuation_handler)
    (body body' : expr) (ts : trap_stack) (ds_out : demands) :
  (∀ ds : demands,
    trap_demands ts body = Some ds ->
    ∃ ds' : demands, trap_demands (m ts) body' = Some ds' ∧ sub ds ds') ->
  (∀ (t : trap_stack) (d : demands),
    handler_trap_demands t h = Some d ->
    ∃ d' : demands, handler_trap_demands (m t) h' = Some d' ∧ sub d d') ->
  trap_demands ts (Let_cont_non_recursive k h body) = Some ds_out ->
  ∃ ds_out' : demands,
    trap_demands (m ts) (Let_cont_non_recursive k h' body') = Some ds_out'
    ∧ sub ds_out ds_out'.
Proof.
  intros Hbody Hh. simpl.
  destruct (among k ts) eqn:Hk; [ discriminate | ].
  rewrite (among_shrinks k ts Hk).
  destruct (trap_demands ts body) as [ds | ] eqn:Hds; [ | discriminate ].
  simpl.
  destruct (Hbody ds eq_refl) as (ds' & Hds' & Hsub). rewrite Hds'. simpl.
  destruct (leave_scope [k] ds) as [outer | ] eqn:Hleave; [ | discriminate ].
  simpl.
  destruct (sub_leave [k] ds ds' outer Hsub Hleave)
    as (outer' & Hleave' & Hsub_outer).
  rewrite Hleave'. simpl.
  pose proof (sub_entry k ds ds' Hsub) as Hentry.
  destruct (entry_of k ds) as [ | entered | ] eqn:Hentry_b.
  - intros [= <-]. rewrite Hentry. exists outer'.
    split; [ reflexivity | exact Hsub_outer ].
  - destruct (among k entered) eqn:Hke; [ discriminate | ].
    destruct (handler_trap_demands entered h) as [dh | ] eqn:Hdh;
      [ | discriminate ].
    simpl. intros [= <-].
    destruct Hentry as [Hentry | Hentry]; rewrite Hentry.
    + exists outer'. split; [ reflexivity | ].
      apply (sub_weaken outer); [ apply incl_appl, incl_refl | ].
      exact Hsub_outer.
    + rewrite (among_shrinks k entered Hke).
      destruct (Hh entered dh Hdh) as (dh' & Hdh' & Hsub_h).
      rewrite Hdh'. simpl. exists (outer' ++ dh').
      split; [ reflexivity | apply sub_app; assumption ].
  - discriminate.
Qed.

End Sub.

(* ======================================================================== *)
(* The passes over the handlers of a recursive group                        *)
(* ======================================================================== *)

(* A pass of the check over the handlers of a recursive [Let_cont], with
   what a handler demands as a parameter [hd]: with
   [handler_trap_demands] it is FlambdaTrapProofs.lax_group_step. *)
Lemma fold_left_nil {A B : Type} (f : A -> B -> A) (init : A) :
  fold_left f init [] = init.
Proof. reflexivity. Qed.

Lemma fold_left_cons {A B : Type} (f : A -> B -> A) (init : A) (x : B)
    (l : list B) :
  fold_left f init (x :: l) = fold_left f (f init x) l.
Proof. reflexivity. Qed.

Section Group.

Variable hd : trap_stack -> continuation_handler -> option demands.

Definition gstep (state : option group_state)
    (kh : continuation * continuation_handler) : option group_state :=
  let '(k, h) := kh in
  let* '(checked, ds) := state in
  if among k checked then Some (checked, ds)
  else
    match entry_of k ds with
    | Not_entered => Some (checked, ds)
    | Entered_with entered =>
      let* ds' := hd entered h in
      Some (k :: checked, ds ++ ds')
    | Entered_with_two_stacks => None
    end.

Definition gpass (handlers : list (continuation * continuation_handler))
    (state : group_state) : option group_state :=
  fold_left gstep (Some state) handlers.

Lemma gfold_none (hs : list (continuation * continuation_handler)) :
  fold_left gstep None hs = None.
Proof.
  induction hs as [ | [k h] hs IH]; simpl; [ reflexivity | exact IH ].
Qed.

(* One handler: the state is as it was, the handler being checked
   already or not entered; or the handler is checked, and its demands
   are added. *)
Lemma gstep_cases (checked : list continuation) (ds : demands)
    (k : continuation) (h : continuation_handler) (state' : group_state) :
  gstep (Some (checked, ds)) (k, h) = Some state' ->
  (state' = (checked, ds)
   ∧ (among k checked = true ∨ entry_of k ds = Not_entered))
  ∨ ∃ (entered : trap_stack) (ds_h : demands),
      among k checked = false
      ∧ entry_of k ds = Entered_with entered
      ∧ hd entered h = Some ds_h
      ∧ state' = (k :: checked, ds ++ ds_h).
Proof.
  simpl. destruct (among k checked) eqn:Hk; [ intros [= <-]; left; auto | ].
  destruct (entry_of k ds) as [ | entered | ] eqn:Hentry;
    [ intros [= <-]; left; auto | | discriminate ].
  destruct (hd entered h) as [ds_h | ] eqn:Hh; [ | discriminate ].
  simpl. intros [= <-]. right. exists entered, ds_h. auto.
Qed.

(* What holds of the handlers checked so far: each was checked under a
   stack that a demand has for it, and what it demands is among the
   demands. *)
Definition ggood (handlers : list (continuation * continuation_handler))
    (state : group_state) : Prop :=
  ∀ k : continuation,
    In k (fst state) ->
    ∃ (h : continuation_handler) (ts : trap_stack) (ds_h : demands),
      In (k, h) handlers
      ∧ In (k, ts) (snd state)
      ∧ hd ts h = Some ds_h
      ∧ incl ds_h (snd state).

Lemma gfold_good (handlers hs : list (continuation * continuation_handler)) :
  incl hs handlers ->
  ∀ state state' : group_state,
    ggood handlers state ->
    fold_left gstep (Some state) hs = Some state' ->
    ggood handlers state' ∧ incl (snd state) (snd state').
Proof.
  induction hs as [ | [k h] hs IH]; intros Hincl [checked ds] state' Hgood;
    rewrite ?fold_left_nil, ?fold_left_cons.
  - intros [= <-]. split; [ exact Hgood | apply incl_refl ].
  - assert (Hhs : incl hs handlers)
      by (intros kh Hin; apply Hincl; right; exact Hin).
    destruct (gstep _ (k, h)) as [state1 | ] eqn:Hstep;
      [ | rewrite gfold_none; discriminate ].
    apply gstep_cases in Hstep
      as [[-> _] | (entered & ds_h & Hk & Hentry & Hh & ->)];
      [ exact (IH Hhs _ _ Hgood) | ].
    intros Hfold.
    assert (Hgood' : ggood handlers (k :: checked, ds ++ ds_h)).
    { intros k' [<- | Hin]; simpl.
      - exists h, entered, ds_h. split; [ apply Hincl; left; reflexivity | ].
        split; [ apply in_or_app; left;
                 exact (entry_of_entered_in k ds entered Hentry) | ].
        split; [ exact Hh | apply incl_appr, incl_refl ].
      - destruct (Hgood k' Hin) as (h' & ts & ds_h' & Hkh & Hd & Hds & Hsub).
        exists h', ts, ds_h'. split; [ exact Hkh | ].
        split; [ apply in_or_app; left; exact Hd | ].
        split; [ exact Hds | apply incl_appl, Hsub ]. }
    destruct (IH Hhs _ _ Hgood' Hfold) as [Hgood'' Hsub].
    split; [ exact Hgood'' | ].
    intros d Hd. apply Hsub. simpl. apply in_or_app. left. exact Hd.
Qed.

Lemma giterate_good (handlers : list (continuation * continuation_handler))
    (n : nat) :
  ∀ state state' : group_state,
    ggood handlers state ->
    iterate n (gpass handlers) state = Some state' ->
    ggood handlers state' ∧ incl (snd state) (snd state').
Proof.
  induction n as [ | n IH]; intros state state' Hgood; simpl.
  - intros [= <-]. split; [ exact Hgood | apply incl_refl ].
  - destruct (gpass handlers state) as [state1 | ] eqn:Hpass;
      [ | discriminate ].
    intros Hiter.
    destruct (gfold_good handlers handlers (incl_refl _) state state1 Hgood
                Hpass) as [Hgood1 Hsub1].
    destruct (IH state1 state' Hgood1 Hiter) as [Hgood' Hsub'].
    split; [ exact Hgood' | exact (incl_tran Hsub1 Hsub') ].
Qed.

(* The handlers checked are handlers of the group, each once. *)
Lemma gfold_checked (handlers hs : list (continuation * continuation_handler)) :
  incl hs handlers ->
  ∀ state state' : group_state,
    fold_left gstep (Some state) hs = Some state' ->
    NoDup (fst state) ∧ incl (fst state) (map fst handlers) ->
    (NoDup (fst state') ∧ incl (fst state') (map fst handlers))
    ∧ length (fst state) <= length (fst state').
Proof.
  induction hs as [ | [k h] hs IH]; intros Hincl [checked ds] state';
    rewrite ?fold_left_nil, ?fold_left_cons.
  - intros [= <-] Hchecked. split; [ exact Hchecked | apply le_n ].
  - assert (Hhs : incl hs handlers)
      by (intros kh Hin; apply Hincl; right; exact Hin).
    destruct (gstep _ (k, h)) as [state1 | ] eqn:Hstep;
      [ | rewrite gfold_none; discriminate ].
    apply gstep_cases in Hstep
      as [[-> _] | (entered & ds_h & Hk & Hentry & Hh & ->)];
      [ exact (IH Hhs _ _) | ].
    intros Hfold [Hnodup Hsubset].
    assert (Hchecked' :
      NoDup (fst (k :: checked, ds ++ ds_h))
      ∧ incl (fst (k :: checked, ds ++ ds_h)) (map fst handlers)).
    { simpl. split.
      - constructor; [ apply among_false; exact Hk | exact Hnodup ].
      - intros k' [<- | Hin]; [ | exact (Hsubset k' Hin) ].
        apply (in_map fst _ (k, h)). apply Hincl. left. reflexivity. }
    destruct (IH Hhs _ _ Hfold Hchecked') as [Hchecked'' Hlength].
    split; [ exact Hchecked'' | ].
    simpl in Hlength |- *. apply le_S_n. apply le_S. exact Hlength.
Qed.

(* A pass leaves the state as it was, every handler being checked or not
   entered; or it checks one more. *)
Lemma gfold_progress
    (handlers hs : list (continuation * continuation_handler)) :
  incl hs handlers ->
  ∀ state state' : group_state,
    fold_left gstep (Some state) hs = Some state' ->
    NoDup (fst state) ∧ incl (fst state) (map fst handlers) ->
    (state' = state
     ∧ ∀ (k : continuation) (h : continuation_handler),
         In (k, h) hs ->
         among k (fst state) = true ∨ entry_of k (snd state) = Not_entered)
    ∨ length (fst state) < length (fst state').
Proof.
  induction hs as [ | [k h] hs IH]; intros Hincl [checked ds] state';
    rewrite ?fold_left_nil, ?fold_left_cons.
  - intros [= <-] _. left. split; [ reflexivity | intros k h [] ].
  - assert (Hhs : incl hs handlers)
      by (intros kh Hin; apply Hincl; right; exact Hin).
    destruct (gstep _ (k, h)) as [state1 | ] eqn:Hstep;
      [ | rewrite gfold_none; discriminate ].
    apply gstep_cases in Hstep
      as [[-> Hsame] | (entered & ds_h & Hk & Hentry & Hh & ->)].
    + intros Hfold Hchecked.
      destruct (IH Hhs _ _ Hfold Hchecked) as [[-> Hrest] | Hgrows];
        [ left | right; exact Hgrows ].
      split; [ reflexivity | ].
      intros k' h' [[= <- <-] | Hin]; [ exact Hsame | exact (Hrest k' h' Hin) ].
    + intros Hfold [Hnodup Hsubset]. right.
      assert (Hchecked' :
        NoDup (fst (k :: checked, ds ++ ds_h))
        ∧ incl (fst (k :: checked, ds ++ ds_h)) (map fst handlers)).
      { simpl. split.
        - constructor; [ apply among_false; exact Hk | exact Hnodup ].
        - intros k' [<- | Hin]; [ | exact (Hsubset k' Hin) ].
          apply (in_map fst _ (k, h)). apply Hincl. left. reflexivity. }
      destruct (gfold_checked handlers hs Hhs _ _ Hfold Hchecked')
        as [_ Hlength].
      simpl in Hlength |- *. exact Hlength.
Qed.

Lemma iterate_fixed {A : Type} (f : A -> option A) (x : A) (n : nat) :
  f x = Some x -> iterate n f x = Some x.
Proof.
  intros Hfixed. induction n as [ | n IH]; simpl; [ reflexivity | ].
  rewrite Hfixed. exact IH.
Qed.

(* After as many passes as there are handlers, every handler of the
   group is checked or is not entered. *)
Lemma giterate_complete
    (handlers : list (continuation * continuation_handler)) (n : nat) :
  ∀ state final : group_state,
    iterate n (gpass handlers) state = Some final ->
    NoDup (fst state) ∧ incl (fst state) (map fst handlers) ->
    (NoDup (fst final) ∧ incl (fst final) (map fst handlers))
    ∧ ((∀ (k : continuation) (h : continuation_handler),
         In (k, h) handlers ->
         among k (fst final) = true ∨ entry_of k (snd final) = Not_entered)
       ∨ length (fst state) + n <= length (fst final)).
Proof.
  induction n as [ | n IH]; intros state final; simpl.
  - intros [= <-] Hchecked. split; [ exact Hchecked | right ].
    rewrite Nat.add_0_r. apply le_n.
  - destruct (gpass handlers state) as [state1 | ] eqn:Hpass;
      [ | discriminate ].
    intros Hiter Hchecked.
    destruct (gfold_checked handlers handlers (incl_refl _) state state1
                Hpass Hchecked) as [Hchecked1 _].
    destruct (gfold_progress handlers handlers (incl_refl _) state state1
                Hpass Hchecked) as [[-> Hstable] | Hgrows].
    + rewrite (iterate_fixed (gpass handlers) state n Hpass) in Hiter.
      injection Hiter as <-. split; [ exact Hchecked | left; exact Hstable ].
    + destruct (IH state1 final Hiter Hchecked1) as [Hfinal [Hstable | Hlen]].
      * split; [ exact Hfinal | left; exact Hstable ].
      * split; [ exact Hfinal | right ].
        rewrite Nat.add_succ_r.
        apply (Nat.le_trans _ (length (fst state1) + n)); [ | exact Hlen ].
        exact
          (proj1
             (Nat.add_le_mono_r (S (length (fst state)))
                (length (fst state1)) n)
             Hgrows).
Qed.

Lemma gcomplete (handlers : list (continuation * continuation_handler))
    (ds0 : demands) (checked : list continuation) (ds : demands) :
  iterate (length handlers) (gpass handlers) ([], ds0) = Some (checked, ds) ->
  ∀ k : continuation,
    In k (map fst handlers) ->
    among k checked = true ∨ entry_of k ds = Not_entered.
Proof.
  intros Hiter k Hk.
  destruct (giterate_complete handlers (length handlers) ([], ds0)
              (checked, ds) Hiter)
    as [[Hnodup Hsubset] [Hstable | Hlen]].
  - split; [ constructor | intros k' [] ].
  - apply in_map_iff in Hk as ([k' h] & <- & Hin).
    exact (Hstable k' h Hin).
  - left. apply among_In. simpl in *.
    assert (Hlen' : length (map fst handlers) <= length checked)
      by (rewrite length_map; exact Hlen).
    exact (NoDup_length_incl Hnodup Hlen' Hsubset k Hk).
Qed.

End Group.

(* The pass over handlers that have each been rewritten is the pass over
   the handlers, with what a rewritten handler demands. *)
Lemma gfold_map (hd : trap_stack -> continuation_handler -> option demands)
    (f : continuation_handler -> continuation_handler)
    (hs : list (continuation * continuation_handler)) :
  ∀ state : option group_state,
    fold_left (gstep hd) state (map (fun kh => (fst kh, f (snd kh))) hs)
    = fold_left (gstep (fun t h => hd t (f h))) state hs.
Proof.
  induction hs as [ | [k h] hs IH]; intros state; cbn [map fst snd];
    [ reflexivity | ].
  rewrite !fold_left_cons. exact (IH _).
Qed.

(* The passes over a group whose handlers have demands below those they
   had, from demands below those the passes started from: they succeed
   where the first did, and end below where the first ended. *)
Section Group_below.

Variable m : trap_stack -> trap_stack.
Variables hd hd' : trap_stack -> continuation_handler -> option demands.
Variable handlers : list (continuation * continuation_handler).
Hypothesis Hdistinct : distinct (map fst handlers) = true.
Hypothesis Hhandlers :
  ∀ (k : continuation) (h : continuation_handler),
    In (k, h) handlers ->
    ∀ (ts : trap_stack) (ds : demands),
      hd ts h = Some ds ->
      ∃ ds' : demands, hd' (m ts) h = Some ds' ∧ sub m ds ds'.
Variable checked_b : list continuation.
Variable ds_b : demands.
Hypothesis Hgood : ggood hd handlers (checked_b, ds_b).
Hypothesis Hfinal :
  ∀ k : continuation,
    In k (map fst handlers) ->
    match entry_of k ds_b with
    | Not_entered => True
    | Entered_with _ => In k checked_b
    | Entered_with_two_stacks => False
    end.

Lemma gfold_below (hs : list (continuation * continuation_handler)) :
  incl hs handlers ->
  ∀ state' : group_state,
    sub m ds_b (snd state') ->
    ∃ state1' : group_state,
      fold_left (gstep hd') (Some state') hs = Some state1'
      ∧ sub m ds_b (snd state1').
Proof.
  induction hs as [ | [k h] hs IH]; intros Hincl [checked' ds'] Hsub;
    rewrite ?fold_left_nil, ?fold_left_cons.
  - eexists. split; [ reflexivity | exact Hsub ].
  - assert (Hhs : incl hs handlers)
      by (intros kh Hin; apply Hincl; right; exact Hin).
    assert (Hkh : In (k, h) handlers) by (apply Hincl; left; reflexivity).
    simpl in Hsub. simpl gstep.
    destruct (among k checked'); [ exact (IH Hhs (checked', ds') Hsub) | ].
    pose proof (sub_entry m k ds_b ds' Hsub) as Hentry.
    pose proof (Hfinal k (in_map fst _ _ Hkh)) as Hfin.
    destruct (entry_of k ds_b) as [ | entered | ] eqn:Hentry_b;
      [ | | contradiction ].
    + rewrite Hentry. exact (IH Hhs (checked', ds') Hsub).
    + destruct Hentry as [Hentry | Hentry]; rewrite Hentry;
        [ exact (IH Hhs (checked', ds') Hsub) | ].
      destruct (Hgood k Hfin) as (h0 & ts0 & ds_h & Hkh0 & Hd & Hh0 & Hsub_h).
      simpl in Hd, Hsub_h.
      pose proof (lookup_of_distinct k h0 handlers Hdistinct Hkh0) as Hl0.
      rewrite (lookup_of_distinct k h handlers Hdistinct Hkh) in Hl0.
      injection Hl0 as <-.
      pose proof (entry_of_entered k ds_b entered Hentry_b (k, ts0) Hd
                    eq_refl) as Hts0.
      simpl in Hts0. subst ts0.
      destruct (Hhandlers k h Hkh entered ds_h Hh0) as (ds_h' & Hh' & Hsub').
      rewrite Hh'. simpl. apply (IH Hhs (k :: checked', ds' ++ ds_h')).
      simpl.
      apply (sub_weaken m (ds_b ++ ds_h));
        [ apply incl_app; [ apply incl_refl | exact Hsub_h ] | ].
      apply sub_app; assumption.
Qed.

Lemma giterate_below (n : nat) :
  ∀ state' : group_state,
    sub m ds_b (snd state') ->
    ∃ final' : group_state,
      iterate n (gpass hd' handlers) state' = Some final'
      ∧ sub m ds_b (snd final').
Proof.
  induction n as [ | n IH]; intros state' Hsub; simpl.
  - eexists. split; [ reflexivity | exact Hsub ].
  - destruct (gfold_below handlers (incl_refl _) state' Hsub)
      as (state1' & Hpass & Hsub1).
    change (gpass hd' handlers state' = Some state1') in Hpass.
    rewrite Hpass. exact (IH state1' Hsub1).
Qed.

End Group_below.

(* A recursive [Let_cont] whose body and whose handlers have demands
   below those they had. *)
Lemma sub_let_cont_recursive (m : trap_stack -> trap_stack)
    (f : continuation_handler -> continuation_handler)
    (handlers : list (continuation * continuation_handler))
    (body body' : expr) (ts : trap_stack) (ds_out : demands) :
  (∀ (k : continuation) (ts : trap_stack), In k (m ts) -> In k ts) ->
  distinct (map fst handlers) = true ->
  (∀ ds : demands,
    trap_demands ts body = Some ds ->
    ∃ ds' : demands, trap_demands (m ts) body' = Some ds' ∧ sub m ds ds') ->
  (∀ (k : continuation) (h : continuation_handler),
    In (k, h) handlers ->
    ∀ (t : trap_stack) (d : demands),
      handler_trap_demands t h = Some d ->
      ∃ d' : demands,
        handler_trap_demands (m t) (f h) = Some d' ∧ sub m d d') ->
  trap_demands ts (Let_cont_recursive handlers body) = Some ds_out ->
  ∃ ds_out' : demands,
    trap_demands (m ts)
      (Let_cont_recursive (map (fun kh => (fst kh, f (snd kh))) handlers)
         body')
    = Some ds_out'
    ∧ sub m ds_out ds_out'.
Proof.
  intros m_shrinks Hdistinct Hbody Hhandlers.
  rewrite !trap_demands_recursive. cbv zeta.
  rewrite map_fst_handlers, length_map.
  destruct (existsb (fun k => among k ts) (map fst handlers)) eqn:Hts;
    [ discriminate | ].
  assert (Hts' :
    existsb (fun k => among k (m ts)) (map fst handlers) = false).
  { destruct (existsb (fun k => among k (m ts)) (map fst handlers)) eqn:Hex;
      [ | reflexivity ].
    apply existsb_exists in Hex as (k & Hk & Hin).
    assert (Hex' : existsb (fun k => among k ts) (map fst handlers) = true).
    { apply existsb_exists. exists k. split; [ exact Hk | ].
      apply among_In. apply among_In in Hin. exact (m_shrinks k ts Hin). }
    congruence. }
  rewrite Hts'.
  destruct (trap_demands ts body) as [ds0 | ] eqn:Hds0; [ | discriminate ].
  simpl.
  destruct (Hbody ds0 eq_refl) as (ds0' & Hds0' & Hsub0). rewrite Hds0'.
  simpl.
  change (lax_group_pass handlers)
    with (gpass handler_trap_demands handlers).
  destruct
    (iterate (length handlers) (gpass handler_trap_demands handlers)
       ([], ds0))
    as [[checked ds] | ] eqn:Hiter; [ | discriminate ].
  simpl.
  destruct
    (forallb
       (fun k =>
          match entry_of k ds with
          | Not_entered => true
          | Entered_with _ => among k checked
          | Entered_with_two_stacks => false
          end)
       (map fst handlers)) eqn:Hfinal; [ | discriminate ].
  intros Hleave.
  destruct (giterate_good handler_trap_demands handlers (length handlers)
              ([], ds0) (checked, ds))
    as [Hgood Hincl0]; [ intros k [] | exact Hiter | ].
  simpl in Hincl0.
  assert (Hfinal_b :
    ∀ k : continuation,
      In k (map fst handlers) ->
      match entry_of k ds with
      | Not_entered => True
      | Entered_with _ => In k checked
      | Entered_with_two_stacks => False
      end).
  { intros k Hk. rewrite forallb_forall in Hfinal. specialize (Hfinal k Hk).
    destruct (entry_of k ds); [ exact I | apply among_In; exact Hfinal | ].
    discriminate. }
  assert (Hpass :
    ∀ state : group_state,
      lax_group_pass (map (fun kh => (fst kh, f (snd kh))) handlers) state
      = gpass (fun t h => handler_trap_demands t (f h)) handlers state).
  { intros state. exact (gfold_map handler_trap_demands f handlers _). }
  rewrite (iterate_same _ _ _ Hpass).
  destruct
    (giterate_below m handler_trap_demands
       (fun t h => handler_trap_demands t (f h)) handlers Hdistinct
       Hhandlers checked ds Hgood Hfinal_b (length handlers) ([], ds0'))
    as ([checked' ds'] & Hiter' & Hsub_final).
  { simpl. exact (sub_weaken m ds0 ds ds0' Hincl0 Hsub0). }
  rewrite Hiter'. simpl in Hsub_final |- *.
  assert (Hfinal' :
    forallb
      (fun k =>
         match entry_of k ds' with
         | Not_entered => true
         | Entered_with _ => among k checked'
         | Entered_with_two_stacks => false
         end)
      (map fst handlers)
    = true).
  { apply forallb_forall. intros k Hk.
    pose proof (sub_entry m k ds ds' Hsub_final) as Hentry.
    pose proof (Hfinal_b k Hk) as Hfin.
    pose proof
      (gcomplete (fun t h => handler_trap_demands t (f h)) handlers ds0'
         checked' ds' Hiter' k Hk) as Hdone.
    destruct (entry_of k ds) as [ | entered | ]; [ | | contradiction ].
    - rewrite Hentry. reflexivity.
    - destruct Hentry as [Hentry | Hentry]; rewrite Hentry;
        [ reflexivity | ].
      destruct Hdone as [Hdone | Hdone]; [ exact Hdone | ].
      rewrite Hentry in Hdone. discriminate. }
  rewrite Hfinal'.
  exact (sub_leave m m_shrinks (map fst handlers) ds ds' ds_out Hsub_final
           Hleave).
Qed.

(* ======================================================================== *)
(* The trap actions that name a handler removed                             *)
(* ======================================================================== *)

(* A stack without [k]. *)
Definition erase (k : continuation) (ts : trap_stack) : trap_stack :=
  filter (fun k' => negb (Pos.eqb k' k)) ts.

Lemma erase_shrinks (k k' : continuation) (ts : trap_stack) :
  In k' (erase k ts) -> In k' ts.
Proof. intros Hin. apply filter_In in Hin as [Hin _]. exact Hin. Qed.

Lemma erase_absent (k : continuation) (ts : trap_stack) :
  ~ In k ts -> erase k ts = ts.
Proof.
  induction ts as [ | k' ts IH]; simpl; [ reflexivity | ].
  intros Hk. destruct (Pos.eqb_spec k' k) as [-> | _]; simpl.
  - contradiction Hk. left. reflexivity.
  - rewrite IH; [ reflexivity | ]. intros Hin. apply Hk. right. exact Hin.
Qed.

Lemma erase_cons_same (k : continuation) (ts : trap_stack) :
  erase k (k :: ts) = erase k ts.
Proof. simpl. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma erase_cons_other (k k' : continuation) (ts : trap_stack) :
  k' <> k -> erase k (k' :: ts) = k' :: erase k ts.
Proof.
  intros Hne. simpl. destruct (Pos.eqb_spec k' k); [ contradiction | ].
  reflexivity.
Qed.

(* A demand with its stack erased is below the demand. *)
Lemma sub_erase_one (k : continuation) (ds : demands) (c : continuation)
    (ts ts' : trap_stack) :
  In (c, ts) ds -> ts' = erase k ts -> sub (erase k) ds [(c, ts')].
Proof.
  intros Hin -> d' [<- | []]. exists (c, ts). auto.
Qed.

Lemma apply_cont_demands_strip (k : continuation) (ts : trap_stack)
    (ac : apply_cont) (ds : demands) :
  apply_cont_demands ts ac = Some ds ->
  ∃ ds' : demands,
    apply_cont_demands (erase k ts) (strip_trap k ac) = Some ds'
    ∧ sub (erase k) ds ds'.
Proof.
  unfold strip_trap, traps, apply_cont_demands.
  destruct ac as [c args [[handler | handler kind] | ]];
    cbn [ac_trap_action ac_continuation ac_args trap_action_handler].
  - (* [Push] *)
    intros [= <-].
    destruct (Pos.eqb_spec k handler) as [<- | Hne];
      cbn [ac_trap_action ac_continuation].
    + eexists. split; [ reflexivity | ].
      apply (sub_erase_one k _ c (k :: ts)); [ right; left; reflexivity | ].
      rewrite erase_cons_same. reflexivity.
    + eexists. split; [ reflexivity | ].
      intros d' [<- | [<- | []]].
      * exists (handler, ts). split; [ left; reflexivity | auto ].
      * exists (c, handler :: ts). split; [ right; left; reflexivity | ].
        split; [ reflexivity | ]. simpl snd.
        rewrite erase_cons_other; [ reflexivity | ].
        intros Heq. apply Hne. symmetry. exact Heq.
  - (* [Pop] *)
    destruct ts as [ | top rest]; [ discriminate | ].
    destruct (Pos.eqb_spec top handler) as [-> | _]; [ | discriminate ].
    intros [= <-].
    destruct (Pos.eqb_spec k handler) as [<- | Hne];
      cbn [ac_trap_action ac_continuation].
    + eexists. split; [ reflexivity | ].
      apply (sub_erase_one k _ c rest); [ left; reflexivity | ].
      rewrite erase_cons_same. reflexivity.
    + rewrite erase_cons_other
        by (intros Heq; apply Hne; symmetry; exact Heq).
      rewrite Pos.eqb_refl. eexists. split; [ reflexivity | ].
      apply (sub_erase_one k _ c rest); [ left; reflexivity | reflexivity ].
  - (* no trap action *)
    intros [= <-]. cbn [ac_trap_action ac_continuation].
    eexists. split; [ reflexivity | ].
    apply (sub_erase_one k _ c ts); [ left; reflexivity | reflexivity ].
Qed.

Lemma apply_demands_strip (k : continuation) (ts : trap_stack)
    (ap : apply_expr) (ds : demands) :
  Pos.eqb k (ap_exn_continuation ap) = false ->
  apply_demands ts ap = Some ds ->
  ∃ ds' : demands,
    apply_demands (erase k ts) ap = Some ds' ∧ sub (erase k) ds ds'.
Proof.
  intros Hk. unfold apply_demands.
  destruct ts as [ | top rest]; [ discriminate | ].
  destruct (Pos.eqb_spec top (ap_exn_continuation ap)) as [-> | _];
    [ | discriminate ].
  intros [= <-]. apply Pos.eqb_neq in Hk.
  assert (Hne : ap_exn_continuation ap <> k)
    by (intros Heq; apply Hk; symmetry; exact Heq).
  rewrite (erase_cons_other k _ rest Hne), Pos.eqb_refl.
  eexists. split; [ reflexivity | ].
  intros d' [<- | [<- | []]].
  - exists (ap_continuation ap, ap_exn_continuation ap :: rest).
    split; [ left; reflexivity | ]. split; [ reflexivity | ]. simpl snd.
    rewrite (erase_cons_other k _ rest Hne). reflexivity.
  - exists (ap_exn_continuation ap, rest).
    split; [ right; left; reflexivity | auto ].
Qed.

Lemma arms_demands_strip (k : continuation) (ts : trap_stack)
    (arms : list (Z * apply_cont)) :
  ∀ ds : demands,
    arms_demands ts arms = Some ds ->
    ∃ ds' : demands,
      arms_demands (erase k ts)
        (map (fun arm => (fst arm, strip_trap k (snd arm))) arms)
      = Some ds'
      ∧ sub (erase k) ds ds'.
Proof.
  induction arms as [ | [i ac] arms IH]; intros ds; simpl.
  - intros [= <-]. exists []. split; [ reflexivity | apply sub_nil ].
  - destruct (apply_cont_demands ts ac) as [ds1 | ] eqn:H1; [ | discriminate ].
    destruct (arms_demands ts arms) as [ds2 | ] eqn:H2; [ | discriminate ].
    simpl. intros [= <-].
    destruct (apply_cont_demands_strip k ts ac ds1 H1) as (ds1' & H1' & Hs1).
    destruct (IH ds2 eq_refl) as (ds2' & H2' & Hs2).
    rewrite H1', H2'. simpl. eexists.
    split; [ reflexivity | apply sub_app; assumption ].
Qed.

(* An expression that follows the discipline under a stack follows it,
   with the trap actions that name [k] removed, under the stack without
   [k]: provided no call of it has [k] as its exception continuation, it
   does not bind [k], and its binders are distinct. What it then demands
   is what it demanded, with [k] off each stack. *)
Fixpoint trap_demands_strip (k : continuation) (e : expr) :
  called_with_exn k e = false ->
  binds_continuation k e = false ->
  binders_distinct e = true ->
  ∀ (ts : trap_stack) (ds : demands),
    trap_demands ts e = Some ds ->
    ∃ ds' : demands,
      trap_demands (erase k ts) (strip_traps k e) = Some ds'
      ∧ sub (erase k) ds ds'
with handler_trap_demands_strip (k : continuation)
    (h : continuation_handler) :
  handler_called_with_exn k h = false ->
  handler_binds_continuation k h = false ->
  handler_binders_distinct h = true ->
  ∀ (ts : trap_stack) (ds : demands),
    handler_trap_demands ts h = Some ds ->
    ∃ ds' : demands,
      handler_trap_demands (erase k ts) (strip_traps_handler k h) = Some ds'
      ∧ sub (erase k) ds ds'.
Proof.
  - destruct e
      as [x kind d body | bindings body | k1 h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros Hcalled Hbinds Hdistinct ts ds.
    + simpl in *.
      exact (trap_demands_strip k body Hcalled Hbinds Hdistinct ts ds).
    + simpl in *. apply andb_true_iff in Hdistinct as [_ Hdistinct].
      destruct (forallb binding_traps_well_formed bindings);
        [ | discriminate ].
      exact (trap_demands_strip k body Hcalled Hbinds Hdistinct ts ds).
    + cbn [strip_traps]. simpl in Hcalled, Hbinds, Hdistinct.
      apply orb_false_iff in Hcalled as [Hcalled_h Hcalled_body].
      apply orb_false_iff in Hbinds as [Hbinds Hbinds_body].
      apply orb_false_iff in Hbinds as [_ Hbinds_h].
      apply andb_true_iff in Hdistinct as [Hd_h Hd_body].
      apply (sub_let_cont (erase k) (erase_shrinks k)).
      * intros ds0.
        exact (trap_demands_strip k body Hcalled_body Hbinds_body Hd_body ts
                 ds0).
      * intros t d.
        exact (handler_trap_demands_strip k h Hcalled_h Hbinds_h Hd_h t d).
    + cbn [strip_traps]. simpl in Hcalled, Hbinds, Hdistinct.
      apply orb_false_iff in Hcalled as [Hcalled_hs Hcalled_body].
      apply orb_false_iff in Hbinds as [Hbinds_hs Hbinds_body].
      apply andb_true_iff in Hdistinct as [Hdistinct Hd_body].
      apply andb_true_iff in Hdistinct as [Hdistinct Hd_hs].
      apply andb_true_iff in Hdistinct as [_ Hd_names].
      apply (sub_let_cont_recursive (erase k) (strip_traps_handler k));
        [ exact (erase_shrinks k) | exact Hd_names | | ].
      * intros ds0.
        exact (trap_demands_strip k body Hcalled_body Hbinds_body Hd_body ts
                 ds0).
      * clear Hd_names.
        assert (Hall :
          Forall
            (fun kh =>
               ∀ (t : trap_stack) (d : demands),
                 handler_trap_demands t (snd kh) = Some d ->
                 ∃ d' : demands,
                   handler_trap_demands (erase k t)
                     (strip_traps_handler k (snd kh))
                   = Some d'
                   ∧ sub (erase k) d d')
            handlers).
        { revert Hcalled_hs Hbinds_hs Hd_hs.
          induction handlers as [ | [k2 h] handlers IH]; simpl;
            [ constructor | ].
          rewrite !orb_false_iff, andb_true_iff.
          intros [Hc_h Hc_rest] [[_ Hb_h] Hb_rest] [Hd_h Hd_rest].
          constructor; [ | exact (IH Hc_rest Hb_rest Hd_rest) ].
          intros t d.
          exact (handler_trap_demands_strip k h Hc_h Hb_h Hd_h t d). }
        intros k2 h Hin. rewrite Forall_forall in Hall.
        exact (Hall (k2, h) Hin).
    + simpl in Hcalled |- *. exact (apply_demands_strip k ts ap ds Hcalled).
    + simpl. exact (apply_cont_demands_strip k ts ac ds).
    + simpl. exact (arms_demands_strip k ts arms ds).
    + simpl. intros [= <-]. exists [].
      split; [ reflexivity | apply sub_nil ].
  - destruct h as [params metadata handler]. simpl.
    intros Hcalled Hbinds Hdistinct.
    apply andb_true_iff in Hdistinct as [_ Hdistinct].
    exact (trap_demands_strip k handler Hcalled Hbinds Hdistinct).
Qed.

Lemma handler_trap_demands_clear (ts : trap_stack)
    (h : continuation_handler) :
  handler_trap_demands ts (clear_exn_handler h) = handler_trap_demands ts h.
Proof. destruct h. reflexivity. Qed.

(* The [Let_cont] of a handler that is demoted: it follows the discipline
   as it did, and demands no more. *)
Lemma trap_demands_demoted (k : continuation) (h : continuation_handler)
    (body : expr) (ts : trap_stack) (ds : demands) :
  called_with_exn k body = false ->
  binds_continuation k body = false ->
  binders_distinct body = true ->
  trap_demands ts (Let_cont_non_recursive k h body) = Some ds ->
  ∃ ds' : demands,
    trap_demands ts
      (Let_cont_non_recursive k (clear_exn_handler h) (strip_traps k body))
    = Some ds'
    ∧ incl ds' ds.
Proof.
  intros Hcalled Hbinds Hdistinct. simpl.
  destruct (among k ts) eqn:Hk; [ discriminate | ].
  destruct (trap_demands ts body) as [ds0 | ] eqn:Hds0; [ | discriminate ].
  simpl.
  destruct (leave_scope [k] ds0) as [outer | ] eqn:Hleave; [ | discriminate ].
  simpl.
  destruct (trap_demands_strip k body Hcalled Hbinds Hdistinct ts ds0 Hds0)
    as (ds0' & Hds0' & Hsub).
  rewrite (erase_absent k ts) in Hds0' by (apply among_false; exact Hk).
  rewrite Hds0'. simpl.
  destruct (sub_leave (erase k) (erase_shrinks k) [k] ds0 ds0' outer Hsub
              Hleave)
    as (outer' & Hleave' & Hsub_outer).
  rewrite Hleave'. simpl.
  assert (Houter : incl outer' outer).
  { intros d' Hin. destruct (Hsub_outer d' Hin) as (d & Hd & Hfst & Hsnd).
    rewrite (erase_absent k (snd d)) in Hsnd
      by (apply (leave_scope_stacks [k] ds0 outer Hleave d Hd); left;
          reflexivity).
    destruct d' as [c' t'], d as [c t]. simpl in Hfst, Hsnd. subst.
    exact Hd. }
  pose proof (sub_entry (erase k) k ds0 ds0' Hsub) as Hentry.
  destruct (entry_of k ds0) as [ | entered | ] eqn:Hentry_b.
  - intros [= <-]. rewrite Hentry. exists outer'.
    split; [ reflexivity | exact Houter ].
  - destruct (among k entered) eqn:Hke; [ discriminate | ].
    destruct (handler_trap_demands entered h) as [dh | ] eqn:Hdh;
      [ | discriminate ].
    simpl. intros [= <-].
    destruct Hentry as [Hentry | Hentry]; rewrite Hentry.
    + exists outer'. split; [ reflexivity | apply incl_appl, Houter ].
    + rewrite (erase_absent k entered) by (apply among_false; exact Hke).
      rewrite Hke, handler_trap_demands_clear, Hdh. simpl.
      exists (outer' ++ dh). split; [ reflexivity | ].
      apply incl_app; [ apply incl_appl, Houter | apply incl_appr, incl_refl ].
  - discriminate.
Qed.

(* ======================================================================== *)
(* The rewrite keeps the discipline of trap actions                         *)
(* ======================================================================== *)

(* Demands that are among others. *)
Abbreviation among_demands := (sub (fun ts : trap_stack => ts)).

Lemma among_demands_refl (ds : demands) : among_demands ds ds.
Proof. intros d' Hin. exists d'. auto. Qed.

Lemma among_demands_incl (ds ds' : demands) :
  among_demands ds ds' -> incl ds' ds.
Proof.
  intros Hsub d' Hin. destruct (Hsub d' Hin) as (d & Hd & Hfst & Hsnd).
  destruct d' as [c' t'], d as [c t]. simpl in Hfst, Hsnd. subst. exact Hd.
Qed.

Lemma incl_among_demands (ds ds' : demands) :
  incl ds' ds -> among_demands ds ds'.
Proof. intros Hincl d' Hin. exists d'. auto. Qed.

Fixpoint trap_demands_demote (e : expr) :
  binders_distinct e = true ->
  ∀ (scope : list continuation) (ts : trap_stack) (ds : demands),
    trap_demands ts e = Some ds ->
    ∃ ds' : demands,
      trap_demands ts (demote_handlers scope e) = Some ds'
      ∧ among_demands ds ds'
with handler_trap_demands_demote (h : continuation_handler) :
  handler_binders_distinct h = true ->
  ∀ (scope : list continuation) (ts : trap_stack) (ds : demands),
    handler_trap_demands ts h = Some ds ->
    ∃ ds' : demands,
      handler_trap_demands ts (demote_handlers_handler scope h) = Some ds'
      ∧ among_demands ds ds'
with binding_traps_demote (binding : static_binding) :
  binding_binders_distinct binding = true ->
  binding_traps_well_formed binding = true ->
  binding_traps_well_formed (demote_handlers_binding binding) = true
with code_traps_demote (c : code) :
  code_binders_distinct c = true ->
  code_traps_well_formed c = true ->
  code_traps_well_formed (demote_handlers_code c) = true.
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros Hdistinct scope ts ds; cbn [demote_handlers];
      try (intros Hds; exists ds;
           split; [ exact Hds | apply among_demands_refl ]).
    + simpl in *. exact (trap_demands_demote body Hdistinct scope ts ds).
    + simpl in *. apply andb_true_iff in Hdistinct as [Hd_bindings Hd_body].
      destruct (forallb binding_traps_well_formed bindings) eqn:Hbindings;
        [ | discriminate ].
      assert (Hbindings' :
        forallb binding_traps_well_formed
          (map demote_handlers_binding bindings)
        = true).
      { revert Hd_bindings Hbindings.
        induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        rewrite !andb_true_iff. intros [Hd Hd_rest] [Hb Hb_rest].
        split; [ exact (binding_traps_demote binding Hd Hb) | ].
        exact (IH Hd_rest Hb_rest). }
      rewrite Hbindings'.
      exact (trap_demands_demote body Hd_body scope ts ds).
    + simpl in Hdistinct. apply andb_true_iff in Hdistinct as [Hd_h Hd_body].
      assert (Hinside :
        trap_demands ts (Let_cont_non_recursive k h body) = Some ds ->
        ∃ ds1 : demands,
          trap_demands ts
            (Let_cont_non_recursive k (demote_handlers_handler scope h)
               (demote_handlers (k :: scope) body))
          = Some ds1
          ∧ among_demands ds ds1).
      { apply (sub_let_cont (fun t => t) (fun _ _ H => H)).
        - intros ds0.
          exact (trap_demands_demote body Hd_body (k :: scope) ts ds0).
        - intros t d. exact (handler_trap_demands_demote h Hd_h scope t d). }
      destruct (demotable scope k h body) eqn:Hdemotable; [ | exact Hinside ].
      intros Hds. destruct (Hinside Hds) as (ds1 & Hds1 & Hsub1).
      apply demotable_true in Hdemotable as (_ & Hcalled & _ & Hbinds).
      destruct
        (trap_demands_demoted k (demote_handlers_handler scope h)
           (demote_handlers (k :: scope) body) ts ds1)
        as (ds2 & Hds2 & Hincl).
      * destruct (called_with_exn k (demote_handlers (k :: scope) body))
          eqn:Hc; [ | reflexivity ].
        apply called_with_exn_demote in Hc. congruence.
      * rewrite binds_continuation_demote. exact Hbinds.
      * rewrite demote_handlers_binders_aux. exact Hd_body.
      * exact Hds1.
      * exists ds2. split; [ exact Hds2 | ].
        apply incl_among_demands.
        exact (incl_tran Hincl (among_demands_incl ds ds1 Hsub1)).
    + simpl in Hdistinct.
      apply andb_true_iff in Hdistinct as [Hdistinct Hd_body].
      apply andb_true_iff in Hdistinct as [Hdistinct Hd_hs].
      apply andb_true_iff in Hdistinct as [_ Hd_names].
      generalize (map fst handlers ++ scope). intros inner.
      apply (sub_let_cont_recursive (fun t => t)
               (demote_handlers_handler inner));
        [ exact (fun _ _ H => H) | exact Hd_names | | ].
      * intros ds0. exact (trap_demands_demote body Hd_body inner ts ds0).
      * clear Hd_names.
        assert (Hall :
          Forall
            (fun kh =>
               ∀ (t : trap_stack) (d : demands),
                 handler_trap_demands t (snd kh) = Some d ->
                 ∃ d' : demands,
                   handler_trap_demands t
                     (demote_handlers_handler inner (snd kh))
                   = Some d'
                   ∧ among_demands d d')
            handlers).
        { revert Hd_hs.
          induction handlers as [ | [k2 h] handlers IH]; simpl;
            [ constructor | ].
          rewrite andb_true_iff. intros [Hd_h Hd_rest].
          constructor; [ | exact (IH Hd_rest) ].
          intros t d. exact (handler_trap_demands_demote h Hd_h inner t d). }
        intros k2 h Hin. rewrite Forall_forall in Hall.
        exact (Hall (k2, h) Hin).
  - destruct h as [params metadata handler]. simpl. intros Hdistinct.
    apply andb_true_iff in Hdistinct as [_ Hdistinct].
    exact (trap_demands_demote handler Hdistinct).
  - destruct binding as [id c | id | s const]; simpl;
      try (intros _; exact (fun H => H)).
    apply code_traps_demote.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    intros Hdistinct. apply andb_true_iff in Hdistinct as [_ Hdistinct].
    destruct (trap_demands [exn] body) as [ds | ] eqn:Hds; [ | discriminate ].
    intros Hmet.
    destruct (trap_demands_demote body Hdistinct [ret; exn] [exn] ds Hds)
      as (ds' & Hds' & Hsub).
    rewrite Hds'. unfold code_demands_met in *.
    rewrite forallb_forall in Hmet |- *. intros d' Hin.
    exact (Hmet d' (among_demands_incl ds ds' Hsub d' Hin)).
Qed.

Theorem demote_handlers_keeps_traps : demote_handlers_keeps_traps_statement.
Proof.
  intros unit_exn scope e Hdistinct. unfold traps_well_formed.
  destruct (trap_demands [unit_exn] e) as [ds | ] eqn:Hds; [ | discriminate ].
  intros Hmet.
  destruct (trap_demands_demote e Hdistinct scope [unit_exn] ds Hds)
    as (ds' & Hds' & Hsub).
  rewrite Hds'. rewrite forallb_forall in Hmet |- *. intros d' Hin.
  exact (Hmet d' (among_demands_incl ds ds' Hsub d' Hin)).
Qed.

(* ======================================================================== *)
(* The machine with a stack of handlers                                     *)
(* ======================================================================== *)

Theorem demote_handlers_stack_from : demote_handlers_stack_from_statement.
Proof.
  intros statics unit_exn scope e b Hcode Hdistinct Htraps.
  assert (Hdistinct' : binders_distinct (demote_handlers scope e) = true)
    by (rewrite demote_handlers_binders; exact Hdistinct).
  pose proof (demote_handlers_keeps_traps unit_exn scope e Hdistinct Htraps)
    as Htraps'.
  pose proof (stack_agrees_from statics unit_exn e b Hcode Hdistinct Htraps)
    as Hbefore.
  pose proof
    (stack_agrees_from statics unit_exn (demote_handlers scope e) b Hcode
       Hdistinct' Htraps')
    as Hafter.
  pose proof
    (demote_handlers_equivalent scope e [] [] statics empty_heap no_reads b)
    as Hsame.
  split; intros H.
  - apply Hbefore, Hsame, Hafter. exact H.
  - apply Hafter, Hsame, Hbefore. exact H.
Qed.

Theorem demote_handlers_stack : demote_handlers_stack_statement.
Proof.
  intros unit_exn scope e b.
  exact (demote_handlers_stack_from empty_static_env unit_exn scope e b
           eq_refl).
Qed.

(* ======================================================================== *)
(* The rewrite commutes with renaming                                       *)
(* ======================================================================== *)

(* The law of the rewrite is proved from the definitions by the tactic
   of FlambdaEquivarianceTactic.v, which states and proves by itself
   the laws of the tests and of the functions that the rewrite is made
   of. Each test compares two names, and a renaming is injective: so
   each answers of the renamed names what it answers of the names.

   The tactic is given one law, in the form that computation leaves:
   the continuations that renamed handlers bind are those that the
   handlers bind, renamed. They are the scope of the handlers of a
   [Let_cont_recursive]. *)
Lemma map_fst_renamed_handlers (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  map
    (fun x : continuation * continuation_handler =>
       fst (let '(k, h) := x in (rename_continuation r k, rename_handler r h)))
    handlers
  = map (fun x => rename_continuation r (fst x)) handlers.
Proof. apply map_ext. intros [k h]. reflexivity. Qed.

#[local] Hint Rewrite map_fst_renamed_handlers : equivariance.

Theorem demote_handlers_equivariant : demote_handlers_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The examples                                                             *)
(* ======================================================================== *)

Theorem demote_handlers_try_raise : demote_handlers_try_raise_statement.
Proof. reflexivity. Qed.

Theorem demotion_under_call : demotion_under_call_statement.
Proof.
  unfold demotion_under_call_statement. split; [ reflexivity | ].
  split; [ reflexivity | ]. split; [ reflexivity | ]. split.
  - exists 20%nat. vm_compute. reflexivity.
  - exists 20%nat. vm_compute. reflexivity.
Qed.
