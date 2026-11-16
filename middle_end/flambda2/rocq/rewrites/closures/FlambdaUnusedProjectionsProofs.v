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

(* Proofs of the statements of FlambdaUnusedProjectionsSpec.v about the
   meaning of the pass. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaClosuresProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaUnusedProjectionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What it means for what is known to hold                                  *)
(* ======================================================================== *)

(* A variable known to be a closure of a set is bound to one, at the
   slot and with the functions that are known. *)
Definition sets_hold (known : known_sets) (vars : env) : Prop :=
  ∀ (x : variable) (slot : function_slot)
    (function_decls : list (function_slot * code_id)),
    lookup x known = Some (slot, function_decls) ->
    ∃ value_slots : list (value_slot * value),
      lookup x vars
      = Some (Value_closure_of_set slot function_decls value_slots).

Lemma lookup_forget_set (x y : variable) (known : known_sets)
    (entry : function_slot * list (function_slot * code_id)) :
  lookup y (forget_set x known) = Some entry ->
  y <> x ∧ lookup y known = Some entry.
Proof.
  unfold forget_set.
  induction known as [ | [z entry'] known IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb_spec x z) as [-> | Hxz]; simpl.
  - intros Hy. destruct (IH Hy) as [Hne Hlookup]. split; [ exact Hne | ].
    destruct (Pos.eqb_spec y z) as [-> | _]; [ contradiction | exact Hlookup ].
  - destruct (Pos.eqb_spec y z) as [-> | Hyz].
    + intros Hy. split; [ congruence | exact Hy ].
    + exact IH.
Qed.

Lemma lookup_forget_sets (xs : list variable) :
  ∀ (y : variable) (known : known_sets)
    (entry : function_slot * list (function_slot * code_id)),
    lookup y (forget_sets xs known) = Some entry ->
    among y xs = false ∧ lookup y known = Some entry.
Proof.
  unfold among.
  induction xs as [ | x xs IH]; intros y known entry; simpl;
    [ intros Hy; split; [ reflexivity | exact Hy ] | ].
  intros Hy. destruct (IH y _ entry Hy) as [Hxs Hforget].
  destruct (lookup_forget_set x y known entry Hforget) as [Hne Hlookup].
  split; [ | exact Hlookup ].
  apply orb_false_iff. split; [ apply Pos.eqb_neq, Hne | exact Hxs ].
Qed.

Lemma sets_hold_nil (vars : env) : sets_hold [] vars.
Proof. intros x slot function_decls Hx. discriminate Hx. Qed.

(* What is known in the body of a [Let] holds once the [Let] has taken
   its step. *)
Lemma sets_hold_under_let (known : known_sets) (vars : env)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record) (x : variable)
      (d : named)
    (v : value) :
  sets_hold known vars ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  sets_hold (sets_under_let known x d) ((x, v) :: vars).
Proof.
  intros Hholds Hexec.
  assert (Hrest : sets_hold (forget_set x known) ((x, v) :: vars)).
  { intros y slot function_decls Hy.
    destruct (lookup_forget_set x y known _ Hy) as [Hne Hlookup].
    simpl. destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | ].
    exact (Hholds y slot function_decls Hlookup). }
  unfold sets_under_let.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hrest.
  apply exec_named_pure in Hexec as [Hd _]; [ | reflexivity ].
  intros y slot function_decls. simpl.
  destruct (Pos.eqb_spec y x) as [-> | Hne].
  - intros [= <- <-]. simpl in Hd.
    destruct (lookup fslot decls) as [code | ]; simpl in Hd;
      [ | discriminate ].
    destruct (eval_value_slots vars slots) as [evaluated | ]; simpl in Hd;
      [ | discriminate ].
    injection Hd as <-. eauto.
  - intros Hy. specialize (Hrest y slot function_decls Hy). simpl in Hrest.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | exact Hrest ].
Qed.

(* What is known of the variables that are not parameters holds in a
   handler. *)
Lemma sets_hold_bind_params (known : known_sets) (params : list variable)
    (values : list value) (vars handler_vars : env) :
  sets_hold known vars ->
  bind_params params values vars = Some handler_vars ->
  sets_hold (forget_sets params known) handler_vars.
Proof.
  intros Hholds Hbind y slot function_decls Hy.
  destruct (lookup_forget_sets params y known _ Hy) as [Hparams Hlookup].
  rewrite (bind_params_lookup_other y params values vars handler_vars
             Hparams Hbind).
  exact (Hholds y slot function_decls Hlookup).
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* Equivalence under the environments in which what is known holds. *)
Definition expr_equivalent_knowing (known : known_sets) (e e' : expr)
    : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    sets_hold known vars ->
    equivalent (State e vars conts statics hp rd)
      (State e' vars conts statics hp rd).

Lemma expr_equivalent_knowing_trans (known : known_sets) (e e' e'' : expr) :
  expr_equivalent_knowing known e e' ->
  expr_equivalent_knowing known e' e'' ->
  expr_equivalent_knowing known e e''.
Proof.
  intros H H' vars conts statics hp rd Hholds.
  exact (equivalent_trans _ _ _ (H vars conts statics hp rd_ Hholds)
    (H' vars conts statics hp rd_ Hholds)).
Qed.

(* A projection that the test accepts has a value, of the kind of a
   closure. *)
Lemma projection_defined_sound (known : known_sets) (vars : env)
    (statics : static_env) (d : named) :
  sets_hold known vars ->
  projection_defined known d = true ->
  ∃ v : value,
    eval_named statics vars d = Some v ∧ kind_of_value v = Value.
Proof.
  intros Hholds.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    simpl; try discriminate.
  destruct closure as [x | sym | c]; try discriminate.
  destruct (lookup x known) as [[slot function_decls] | ] eqn:Hx;
    [ | discriminate ].
  intros Hdefined.
  apply andb_true_iff in Hdefined as [Hdefined Hto].
  apply andb_true_iff in Hdefined as [Hslot Hfrom].
  apply Pos.eqb_eq in Hslot. subst slot.
  apply is_bound_lookup in Hfrom as [code Hfrom].
  apply is_bound_lookup in Hto as [code' Hto].
  destruct (Hholds x move_from function_decls Hx) as [value_slots Hvalue].
  exists (Value_closure_of_set move_to function_decls value_slots).
  split; [ | reflexivity ].
  unfold eval_named. cbn [eval_simple]. rewrite Hvalue.
  exact
    (project_function_slot_of_set statics move_from move_to function_decls
       value_slots code code' Hfrom Hto).
Qed.

Lemma let_equivalent_knowing (known : known_sets) (x : variable)
    (kind : flambda_kind) (d : named) (body body' : expr) :
  expr_equivalent_knowing (sets_under_let known x d) body body' ->
  expr_equivalent_knowing known (Let x kind d body) (Let x kind d body').
Proof.
  intros Hbody vars conts statics hp rd Hholds.
  apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' Hd _ ].
  apply Hbody.
  exact (sets_hold_under_let known vars statics hp hp' rd rd' x d v Hholds Hd).
Qed.

(* The projection has a value of the kind of the [Let], so the [Let]
   takes its step; the body does not look at the variable. *)
Lemma unused_projection_let (known : known_sets) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  kind_eqb kind Value = true ->
  projection_defined known d = true ->
  ~ var_free_in_expr x body ->
  expr_equivalent_knowing known (Let x kind d body) body.
Proof.
  intros Hkind Hdefined Hunused vars conts statics hp rd Hholds.
  destruct (projection_defined_sound known vars statics d Hholds Hdefined)
    as (v & Hd & Hv).
  apply kind_eqb_eq in Hkind. subst kind.
  assert (Hkind : kind_eqb (kind_of_value v) Value = true)
    by (apply kind_eqb_eq, Hv).
  assert (Hnot : ∀ (p : prim) (args : list simple), d <> Prim p args).
  { intros p args ->. discriminate Hdefined. }
  apply (equivalent_trans _ (State body ((x, v) :: vars) conts statics hp rd));
    [ apply step_equivalent;
      rewrite (step_let_not_prim _ _ _ _ _ _ _ _ rd_ Hnot), Hd, Hkind;
      reflexivity
    | ].
  apply coincide_equivalent.
  - intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | reflexivity ].
  - intros k _. apply (conts_agree_same identity_renaming). reflexivity.
Qed.

Lemma let_cont_handler_equivalent_knowing (known : known_sets)
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr) :
  expr_equivalent_knowing (forget_sets params known) handler handler' ->
  expr_equivalent_knowing known
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  intros Hhandler vars conts statics hp rd Hholds.
  apply refines_antisymmetric; apply let_cont_handler_refines_at;
    intros values handler_vars conts' statics' hp' rd' _ Hbind;
    apply equivalent_refines; [ | apply equivalent_sym ];
    apply Hhandler;
    exact (sets_hold_bind_params known params values vars handler_vars
             Hholds Hbind).
Qed.

Fixpoint remove_projections_knowing_equivalent (e : expr) :
  ∀ known : known_sets,
    expr_equivalent_knowing known e (remove_projections_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros known; simpl;
    try (intros vars conts statics hp rd _; apply equivalent_refl).
  - pose proof (let_equivalent_knowing known x kind d _ _
      (remove_projections_knowing_equivalent body (sets_under_let known x d)))
      as Hbody.
    destruct
      (kind_eqb kind Value
       && projection_defined known d
       && negb
            (var_free_b x
               (remove_projections_knowing (sets_under_let known x d) body)))
      eqn:Hremove; [ | exact Hbody ].
    apply andb_true_iff in Hremove as [Hdefined Hunused].
    apply andb_true_iff in Hdefined as [Hkind Hdefined].
    apply negb_true_iff in Hunused.
    apply (expr_equivalent_knowing_trans known _ _ _ Hbody),
      unused_projection_let; [ exact Hkind | exact Hdefined | ].
    intros Hx. apply var_free_b_sound in Hx. congruence.
  - intros vars conts statics hp rd Hholds.
    apply let_static_equivalent_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    apply (remove_projections_knowing_equivalent body known), Hholds.
  - apply (expr_equivalent_knowing_trans known _
      (Let_cont_non_recursive k
         (Handler params metadata
            (remove_projections_knowing (forget_sets params known) handler))
         body)).
    + apply let_cont_handler_equivalent_knowing,
        remove_projections_knowing_equivalent.
    + intros vars conts statics hp rd Hholds.
      apply let_cont_non_recursive_equivalent_step.
      apply (remove_projections_knowing_equivalent body known), Hholds.
  - intros vars conts statics hp rd Hholds.
    apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
    apply (remove_projections_knowing_equivalent body known), Hholds.
Qed.

Theorem remove_unused_projections_equivalent :
  remove_unused_projections_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (remove_projections_knowing_equivalent e []), sets_hold_nil.
Qed.
