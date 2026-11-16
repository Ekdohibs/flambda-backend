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

(* Proofs of the statements of FlambdaShareProjectionsSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaNominalSpec
  FlambdaNominalProofs
  FlambdaApplyProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaCseSpec
  FlambdaCseProofs FlambdaShareProjectionsSpec FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* What it means for what is known to hold                                  *)
(* ======================================================================== *)

(* A variable known to hold a projection is bound to the value of the
   projection. *)
Definition projected_holds (statics : static_env) (vars : env)
    (entry : variable * named) : Prop :=
  is_projection (snd entry) = true
  ∧ ∃ v : value,
      eval_named statics vars (snd entry) = Some v
      ∧ lookup (fst entry) vars = Some v.

Definition all_projected (known : projected) (vars : env)
    (statics : static_env) : Prop :=
  ∀ entry : variable * named,
    entry ∈ known -> projected_holds statics vars entry.

(* ======================================================================== *)
(* The value of a projection                                                *)
(* ======================================================================== *)

(* Binding a variable that the closure is not. *)
Lemma eval_projection_extend (statics : static_env) (vars : env)
    (x : variable) (v : value) (d : named) :
  is_projection d = true ->
  var_in_simples x (projected_from d) = false ->
  eval_named statics ((x, v) :: vars) d = eval_named statics vars d.
Proof.
  intros Hprojection Hmention.
  assert (Hclosure :
    ∀ closure : simple,
      projected_from d = [closure] ->
      eval_simple ((x, v) :: vars) closure = eval_simple vars closure).
  { intros closure Hfrom. apply eval_simple_extend. intros Hfree.
    assert (Hin : var_in_simples x [closure] = true).
    { apply var_in_simples_sound. apply Exists_cons_hd. exact Hfree. }
    rewrite Hfrom in Hmention. congruence. }
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try discriminate Hprojection; simpl;
    rewrite (Hclosure closure eq_refl); reflexivity.
Qed.

(* A value as a closure of a set, under a static environment that
   binds more. *)
Lemma closure_in_set_statics_extend (statics statics' : static_env)
    (v : value)
    (closure :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value)) :
  statics_extend statics statics' ->
  closure_in_set statics v = Some closure ->
  closure_in_set statics' v = Some closure.
Proof.
  intros [_ Hvalues]. destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) eqn:Hlookup;
      [ rewrite (Hvalues _ _ Hlookup); exact (fun H => H) | discriminate ]
  end.
Qed.

(* A projection has the value it had. *)
Lemma eval_projection_statics_extend (statics statics' : static_env)
    (vars : env) (d : named) (v : value) :
  statics_extend statics statics' ->
  is_projection d = true ->
  eval_named statics vars d = Some v ->
  eval_named statics' vars d = Some v.
Proof.
  intros Hextend Hprojection.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try discriminate Hprojection; simpl;
    destruct (eval_simple vars closure) as [c | ]; simpl;
    try discriminate.
  - destruct (closure_of statics c) as [called | ] eqn:Hcalled;
      [ | discriminate ].
    rewrite (closure_of_statics_extend statics statics' c
               (fun s allocated _ => proj2 Hextend s allocated)
               called Hcalled).
    exact (fun H => H).
  - unfold project_value_slot.
    destruct (closure_in_set statics c) as [in_set | ] eqn:Hin_set;
      [ | discriminate ].
    rewrite (closure_in_set_statics_extend statics statics' c in_set
               Hextend Hin_set).
    exact (fun H => H).
Qed.

(* ======================================================================== *)
(* What is known stays so                                                   *)
(* ======================================================================== *)

(* Binding a variable, once what mentions it is forgotten. *)
Lemma all_projected_forget (known : projected) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  all_projected known vars statics ->
  all_projected (forget_projected x known) ((x, v) :: vars) statics.
Proof.
  intros Hhold entry Hin. unfold forget_projected in Hin.
  apply filter_In in Hin. destruct Hin as [Hin Hkept]. cbv beta in Hkept.
  apply andb_true_iff in Hkept as [Hkey Hfrom].
  apply negb_true_iff, Pos.eqb_neq in Hkey. apply negb_true_iff in Hfrom.
  destruct (Hhold entry Hin) as (Hprojection & r & Hvalue & Hlookup).
  split; [ exact Hprojection | ]. exists r. split.
  - rewrite eval_projection_extend; assumption.
  - simpl. destruct (Pos.eqb_spec (fst entry) x) as [Heq | _];
      [ | exact Hlookup ].
    exfalso. apply Hkey. symmetry. exact Heq.
Qed.

Lemma all_projected_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (known : projected) (vs : list value) (vars handler_vars : env),
    all_projected known vars statics ->
    bind_params params vs vars = Some handler_vars ->
    all_projected (forget_projected_vars params known) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, all_projected_forget, Hhold.
Qed.

(* A static environment that binds more. *)
Lemma all_projected_extend (known : projected) (vars : env)
    (statics statics' : static_env) :
  all_projected known vars statics -> statics_extend statics statics' ->
  all_projected known vars statics'.
Proof.
  intros Hhold Hextend entry Hin.
  destruct (Hhold entry Hin) as (Hprojection & v & Hvalue & Hlookup).
  split; [ exact Hprojection | ]. exists v.
  split; [ | exact Hlookup ].
  exact (eval_projection_statics_extend statics statics' vars (snd entry) v
           Hextend Hprojection Hvalue).
Qed.

(* Under a [Let]. *)
Lemma all_projected_under_let (known : projected) (x : variable) (d : named)
    (vars : env) (statics : static_env) (v : value) :
  all_projected known vars statics ->
  eval_named statics vars d = Some v ->
  all_projected (projected_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd. unfold projected_under_let. cbv zeta.
  pose proof (all_projected_forget known vars statics x v Hhold) as Hforgot.
  destruct (is_projection d && negb (var_in_simples x (projected_from d)))
    eqn:Hadd;
    [ | exact Hforgot ].
  apply andb_true_iff in Hadd as [Hprojection Hmention].
  apply negb_true_iff in Hmention.
  intros entry [<- | Hin]; [ | exact (Hforgot entry Hin) ].
  split; [ exact Hprojection | ]. exists v. simpl. split.
  - rewrite eval_projection_extend; assumption.
  - rewrite Pos.eqb_refl. reflexivity.
Qed.

(* ======================================================================== *)
(* Sharing                                                                  *)
(* ======================================================================== *)

Lemma same_simples_sound (args args' : list simple) :
  same_simples args args' = true -> args = args'.
Proof.
  unfold same_simples.
  destruct (list_eq_dec simple_eq_dec args args') as [Heq | ];
    [ intros _; exact Heq | discriminate ].
Qed.

Lemma same_projection_sound (d d' : named) :
  same_projection d d' = true -> d = d'.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    destruct d' as [s' | p' args' | code' slots' | fslot' decls' slots'
      | slot' closure' | pfrom' slot' closure'
      | move_from' move_to' closure'];
    simpl; try discriminate.
  - intros Hsame. apply andb_true_iff in Hsame as [Hslot Hclosure].
    apply Pos.eqb_eq in Hslot. apply same_simples_sound in Hclosure.
    injection Hclosure as ->. subst. reflexivity.
  - intros Hsame. apply andb_true_iff in Hsame as [Hsame Hclosure].
    apply andb_true_iff in Hsame as [Hfrom Hslot].
    apply Pos.eqb_eq in Hfrom, Hslot. apply same_simples_sound in Hclosure.
    injection Hclosure as ->. subst. reflexivity.
Qed.

Lemma projected_by_sound (known : projected) (d : named) (x : variable) :
  projected_by known d = Some x -> (x, d) ∈ known.
Proof.
  induction known as [ | [y d'] known IH]; simpl; [ discriminate | ].
  destruct (same_projection d d') eqn:Hsame.
  - intros [= ->]. apply same_projection_sound in Hsame. subst.
    left. reflexivity.
  - intros Hprojected. right. apply IH, Hprojected.
Qed.

Lemma share_projection_sound (known : projected) (vars : env)
    (statics : static_env) (d : named) (v : value) :
  all_projected known vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (share_projection known d) = Some v.
Proof.
  intros Hhold Hd. unfold share_projection.
  destruct (projected_by known d) as [x | ] eqn:Hprojected; [ | exact Hd ].
  destruct (Hhold _ (projected_by_sound _ _ _ Hprojected))
    as (_ & r & Hvalue & Hlookup).
  simpl in Hvalue, Hlookup. rewrite Hd in Hvalue. injection Hvalue as <-.
  simpl. exact Hlookup.
Qed.

(* ======================================================================== *)
(* With the run of the machine                                              *)
(* ======================================================================== *)

(* The machine runs a [Let] with the heap ([exec_named]); the two facts
   above are about the value alone ([eval_named]). A projection is not a
   primitive, so the machine runs it without the heap and the two agree;
   of what is not a projection the pass records nothing and changes
   nothing. *)
Lemma is_projection_reads_no_heap (d : named) :
  is_projection d = true -> named_reads_heap d = false.
Proof.
  destruct d; simpl; intros Hprojection;
    first [ reflexivity | discriminate Hprojection ].
Qed.

Lemma all_projected_under_let_exec (known : projected) (x : variable)
    (d : named) (vars : env) (statics : static_env) (hp hp' : heap)
      (rd rd' : read_record)
    (v : value) :
  all_projected known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  all_projected (projected_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd. destruct (is_projection d) eqn:Hprojection.
  - apply all_projected_under_let; [ exact Hhold | ].
    exact (proj1 (exec_named_pure statics hp hp' rd rd' vars d v
      (is_projection_reads_no_heap d Hprojection) Hd)).
  - unfold projected_under_let. cbv zeta. rewrite Hprojection. simpl.
    exact (all_projected_forget known vars statics x v Hhold).
Qed.

Lemma share_projection_exec (known : projected) (vars : env)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record) (d : named)
      (v : value) :
  all_projected known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars (share_projection known d) = Some (v, hp', rd').
Proof.
  intros Hhold Hd. unfold share_projection.
  destruct (projected_by known d) as [x | ] eqn:Hprojected; [ | exact Hd ].
  destruct (Hhold _ (projected_by_sound _ _ _ Hprojected))
    as (Hprojection & r & Hvalue & Hlookup).
  simpl in Hprojection, Hvalue, Hlookup.
  destruct (exec_named_pure statics hp hp' rd rd' vars d v
    (is_projection_reads_no_heap d Hprojection) Hd) as (Heval & -> & ->).
  rewrite Heval in Hvalue. injection Hvalue as <-.
  simpl. rewrite Hlookup. reflexivity.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* As in FlambdaCseProofs.v. *)
Fixpoint share_projections_knowing_refines (e : expr) :
  ∀ (known : projected) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    all_projected known vars statics ->
    refines (State e vars conts statics hp rd)
      (State (share_projections_knowing known e) vars conts statics hp rd)
with share_projections_in_handler_refines (h : continuation_handler) :
  ∀ (known : projected) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    all_projected known vars statics ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (share_projections_in_handler known h)
            body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let: the machine runs [d] with the heap; the pass runs the
         same or a variable that holds its value, and leaves the same
         heap. *)
      apply let_refines_step_forward.
      * intros v hp' rd' Hd.
        exact (share_projection_exec known vars statics hp hp' rd rd' d v Hhold
          Hd).
      * intros v hp' rd' Hd _. apply share_projections_knowing_refines.
        exact (all_projected_under_let_exec known x d vars statics hp hp' rd rd'
          v Hhold Hd).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply share_projections_knowing_refines.
      exact (all_projected_extend known vars statics extended Hhold
        (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (share_projections_in_handler known h)
              body)
           vars conts statics hp rd)).
      * apply share_projections_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          share_projections_knowing_refines, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply share_projections_knowing_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply (let_cont_handler_refines_where (statics_extend statics)
      (statics_extend_trans statics));
      [ apply statics_extend_refl | ].
    intros vs handler_vars conts' statics' hp' rd' Hextend _ Hbind.
    apply share_projections_knowing_refines.
    apply (all_projected_bind_params statics' params known vs vars
      handler_vars);
      [ exact (all_projected_extend known vars statics statics' Hhold
          Hextend)
      | exact Hbind ].
Qed.

Theorem share_known_projections_refines :
  share_known_projections_refines_statement.
Proof.
  intros e vars conts statics hp rd. apply share_projections_knowing_refines.
  intros entry [].
Qed.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* The two theorems are proved from the definitions ([equivariance],
   of FlambdaEquivarianceTactic.v), which derives the laws of the
   functions that the pass calls. Three things are given to it, for
   this file only.

   An instance says that [rename_projected] renames the type
   [projected].

   The test of two projections compares two lists of one simple each
   ([same_simples [closure] [closure']]). The law that
   FlambdaCseProofs.v has of that test is of two mapped lists, and
   does not apply to a list that is written out: it is given here for
   two lists of one element.

   The law of [forget_projected] is stated and given to the database.
   Without it the tactic derives that law and still leaves a goal in
   the law of [forget_projected_vars], where it has made an induction
   inside another. *)

#[local] Instance projected_renaming : Renaming_of projected :=
  rename_projected.

Lemma same_simples_of_one_renamed (r : renaming) (s s' : simple) :
  same_simples [rename_simple r s] [rename_simple r s']
  = same_simples [s] [s'].
Proof. exact (same_simples_renamed r [s] [s']). Qed.
#[local] Hint Rewrite same_simples_of_one_renamed : equivariance.

Lemma forget_projected_renamed (r : renaming) (x : variable)
    (known : projected) :
  forget_projected (rename_variable r x) (rename_projected r known)
  = rename_projected r (forget_projected x known).
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_projected_renamed : equivariance.

Theorem share_projections_knowing_equivariant :
  share_projections_knowing_equivariant_statement.
Proof. equivariance. Qed.

Theorem share_known_projections_equivariant :
  share_known_projections_equivariant_statement.
Proof. intros r e. exact (share_projections_knowing_equivariant r [] e). Qed.
