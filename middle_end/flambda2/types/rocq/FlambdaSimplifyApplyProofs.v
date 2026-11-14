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


(* Proofs of the statements of FlambdaSimplifyApplySpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaApplySpec
  FlambdaApplyProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaEnvRenameSpec FlambdaEnvRenameProofs FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaTypingEnvRefinementProofs
  FlambdaProvers FlambdaProversSpec FlambdaProversProofs
  FlambdaProversRenameSpec FlambdaProversRenameProofs
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPrimitivesProofs
  FlambdaSimplifyProjectionsSpec FlambdaSimplifyProjectionsProofs
  FlambdaSimplifyApplySpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* From the type of the callee to its code                                  *)
(* ======================================================================== *)

Theorem callee_code_proved : callee_code_proved_statement.
Proof.
  intros tenv statics vars callee ty slot alloc_mode entry function_type v
    Hsat Hty Hproved Heval.
  destruct (argument_type_satisfied tenv statics vars callee ty v Hsat Hty
              Heval)
    as [valuation [Hsatisfied [_ Htyped]]].
  exact (single_closures_entry_code statics valuation slot alloc_mode entry
           function_type v
           (prove_single_closures_entry_sound statics valuation tenv ty
              (slot, alloc_mode, entry, function_type) v Hsatisfied I Htyped
              Hproved)).
Qed.

(* ======================================================================== *)
(* The call                                                                 *)
(* ======================================================================== *)

Lemma with_call_kind_same (ap : apply_expr) :
  with_call_kind ap (ap_call_kind ap) = ap.
Proof. destruct ap. reflexivity. Qed.

Theorem apply_direct_proved : apply_direct_proved_statement.
Proof.
  intros tenv ap ty slot alloc_mode entry function_type Hkind Hty Hproved
    vars conts statics hp rd Hsat.
  destruct (eval_simple vars (ap_callee ap)) as [v | ] eqn:Heval.
  - destruct (callee_code_proved tenv statics vars (ap_callee ap) ty slot
                alloc_mode entry function_type v Hsat Hty Hproved Heval)
      as [value_slots Hclosure].
    destruct ap as [callee kind args k exn extra args_arity return_arity].
    simpl in Hkind, Heval. subst kind.
    exact (apply_indirect_known callee args extra k exn args_arity
             return_arity v (function_type_code_id function_type)
             value_slots vars conts statics _ rd_ Heval Hclosure).
  - apply step_eq_equivalent. rewrite !step_apply. simpl. rewrite Heval.
    reflexivity.
Qed.

Theorem apply_canonical : apply_canonical_statement.
Proof.
  intros tenv ap callee args Hcallee Hargs vars conts statics hp rd Hsat.
  apply step_eq_equivalent. rewrite !step_apply. simpl.
  rewrite (canonical_in_terms_sound tenv statics vars (ap_callee ap) callee
             Hsat Hcallee).
  rewrite (canonical_in_terms_simples tenv statics vars (ap_args ap) args
             Hsat Hargs).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

Lemma direct_call_kind_equivalent (tenv : Typing_env.t) (ap : apply_expr) :
  expr_equivalent_under tenv (Apply ap)
    (Apply (with_call_kind ap (direct_call_kind tenv ap))).
Proof.
  unfold direct_call_kind.
  destruct (ap_call_kind ap) as [code | ] eqn:Hkind.
  - rewrite <- Hkind, with_call_kind_same. intros vars conts statics hp rd _.
    apply equivalent_refl.
  - destruct (simplify_simple tenv (ap_callee ap)) as [[ty canonical] | ]
      eqn:Hcallee.
    + destruct (simplify_simple_answer tenv (ap_callee ap) canonical ty
                  Hcallee)
        as [Hty _].
      destruct (Provers.prove_single_closures_entry tenv ty)
        as [[[[[slot alloc_mode] entry] function_type] | ] | ] eqn:Hproved.
      * exact (apply_direct_proved tenv ap ty slot alloc_mode entry
                 function_type Hkind Hty Hproved).
      * rewrite <- Hkind, with_call_kind_same. intros vars conts statics hp rd
        _.
        apply equivalent_refl.
      * rewrite <- Hkind, with_call_kind_same. intros vars conts statics hp rd
        _.
        apply equivalent_refl.
    + rewrite <- Hkind, with_call_kind_same. intros vars conts statics hp rd _.
      apply equivalent_refl.
Qed.

Lemma canonical_apply_equivalent (tenv : Typing_env.t) (ap : apply_expr) :
  expr_equivalent_under tenv (Apply ap) (Apply (canonical_apply tenv ap)).
Proof.
  unfold canonical_apply.
  destruct (simplify_simple tenv (ap_callee ap)) as [[ty callee] | ]
    eqn:Hcallee;
    [ | intros vars conts statics hp rd _; apply equivalent_refl ].
  destruct (simplify_simples tenv (ap_args ap)) as [answers | ] eqn:Hargs;
    [ | intros vars conts statics hp rd _; apply equivalent_refl ].
  apply apply_canonical.
  - exact (proj2 (simplify_simple_answer tenv (ap_callee ap) callee ty
                    Hcallee)).
  - exact (proj2 (simplify_simples_answers tenv (ap_args ap) answers Hargs)).
Qed.

Theorem simplify_apply_equivalent_under
  : simplify_apply_equivalent_under_statement.
Proof.
  intros tenv ap vars conts statics hp rd Hsat. unfold simplify_apply.
  apply (equivalent_trans _
           (State (Apply (with_call_kind ap (direct_call_kind tenv ap))) vars
              conts statics hp rd)).
  - exact (direct_call_kind_equivalent tenv ap vars conts statics hp rd_ Hsat).
  - exact (canonical_apply_equivalent tenv
             (with_call_kind ap (direct_call_kind tenv ap)) vars conts
             statics _ rd_ Hsat).
Qed.

Theorem simplify_apply_refines_under
  : simplify_apply_refines_under_statement.
Proof.
  intros tenv ap.
  exact (expr_equivalent_under_refines tenv (Apply ap)
           (simplify_apply tenv ap)
           (simplify_apply_equivalent_under tenv ap)).
Qed.

(* Renaming. *)
Lemma simplify_simple_rename (r : renaming) (tenv : Typing_env.t)
    (s : simple) :
  simplify_simple (rename_typing_env r tenv) (rename_simple r s)
  = option_map
      (fun answer => (rename_type r (fst answer), rename_simple r (snd answer)))
      (simplify_simple tenv s).
Proof.
  unfold simplify_simple.
  rewrite (mem_simple_equivariant r (Some Name_mode.normal) tenv s).
  rewrite (type_simple_in_term_exn_equivariant r tenv
             (Some Name_mode.normal) s).
  destruct (Typing_env.mem_simple (Some Name_mode.normal) tenv s);
    [ | reflexivity ].
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty canonical] | ] | ]; reflexivity.
Qed.

Lemma simplify_simples_rename (r : renaming) (tenv : Typing_env.t)
    (args : list simple) :
  simplify_simples (rename_typing_env r tenv) (map (rename_simple r) args)
  = option_map
      (map
         (fun answer =>
            (rename_type r (fst answer), rename_simple r (snd answer))))
      (simplify_simples tenv args).
Proof.
  induction args as [ | arg args IH]; [ reflexivity | ].
  simpl. rewrite simplify_simple_rename, IH.
  destruct (simplify_simple tenv arg) as [answer | ]; [ | reflexivity ].
  destruct (simplify_simples tenv args) as [answers | ]; reflexivity.
Qed.

Lemma direct_call_kind_rename (r : renaming) (tenv : Typing_env.t)
    (ap : apply_expr) :
  direct_call_kind (rename_typing_env r tenv) (rename_apply r ap)
  = direct_call_kind tenv ap.
Proof.
  unfold direct_call_kind. simpl.
  destruct (ap_call_kind ap) as [code | ]; [ reflexivity | ].
  rewrite simplify_simple_rename.
  destruct (simplify_simple tenv (ap_callee ap)) as [[ty canonical] | ];
    [ | reflexivity ].
  simpl. rewrite (prove_single_closures_entry_rename r tenv ty).
  destruct (Provers.prove_single_closures_entry tenv ty)
    as [[[[[slot alloc_mode] entry] function_type] | ] | ]; reflexivity.
Qed.

Theorem simplify_apply_rename : simplify_apply_rename_statement.
Proof.
  intros r tenv ap. unfold simplify_apply. simpl. f_equal.
  unfold canonical_apply. simpl.
  rewrite direct_call_kind_rename, simplify_simple_rename,
    simplify_simples_rename.
  destruct (simplify_simple tenv (ap_callee ap)) as [[ty callee] | ];
    [ | reflexivity ].
  destruct (simplify_simples tenv (ap_args ap)) as [answers | ];
    [ | reflexivity ].
  simpl. unfold with_callee_and_args, rename_apply. simpl.
  rewrite !map_map. reflexivity.
Qed.

(* ======================================================================== *)
(* A closure that is made                                                   *)
(* ======================================================================== *)

Lemma eval_value_slots_names (vars : env)
    (value_slots : list (value_slot * simple)) :
  ∀ vs : list (value_slot * value),
    eval_value_slots vars value_slots = Some vs ->
    map fst vs = map fst value_slots.
Proof.
  induction value_slots as [ | [slot s] value_slots IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ]; [ | discriminate ]. simpl.
    destruct (eval_value_slots vars value_slots) as [vs' | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl. f_equal. exact (IH vs' eq_refl).
Qed.

Lemma map_fst_combine_same {A B C : Type} (l : list (A * C)) :
  ∀ l' : list B,
    length l' = length l -> map fst (combine (map fst l) l') = map fst l.
Proof.
  induction l as [ | a l IH]; intros [ | b l']; simpl; intros Hlength;
    try reflexivity; try discriminate Hlength.
  f_equal. apply IH. injection Hlength as Hlength. exact Hlength.
Qed.

(* The value of a slot has the type that the list of types has for the
   slot. *)
Lemma value_slot_typed (tenv : Typing_env.t) (statics : static_env)
    (vars : env) :
  env_satisfies tenv statics vars ->
  ∀ (value_slots : list (value_slot * simple)) (tys : list flambda_type)
    (vs : list (value_slot * value)),
    Forall2 (argument_type tenv) (map snd value_slots) tys ->
    eval_value_slots vars value_slots = Some vs ->
    ∀ (w : value_slot) (ty : flambda_type) (contents : value),
      lookup w (combine (map fst value_slots) tys) = Some ty ->
      lookup w vs = Some contents ->
      type_satisfied statics vars contents ty.
Proof.
  intros Hsat.
  induction value_slots as [ | [slot s] value_slots IH];
    intros tys vs Htys Hvs w ty contents Hty Hcontents.
  - discriminate Hty.
  - simpl in Htys. inversion Htys as [ | s' ty' args tys' Hs Hrest]; subst.
    simpl in Hvs.
    destruct (eval_simple vars s) as [v | ] eqn:Hv; [ | discriminate ].
    simpl in Hvs.
    destruct (eval_value_slots vars value_slots) as [vs' | ] eqn:Hvs';
      [ | discriminate ].
    simpl in Hvs. injection Hvs as <-.
    simpl in Hty, Hcontents.
    destruct (Pos.eqb w slot).
    + injection Hty as <-. injection Hcontents as <-.
      exact (argument_type_in_machine tenv statics vars s ty' v Hsat Hs Hv).
    + exact (IH tys' vs' Hrest eq_refl w ty contents Hty Hcontents).
Qed.

Lemma closure_of_set_type_satisfied (tenv : Typing_env.t)
    (statics : static_env) (vars : env) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (tys : list flambda_type)
    (vs : list (value_slot * value)) (v : value) :
  env_satisfies tenv statics vars ->
  Forall2 (argument_type tenv) (map snd value_slots) tys ->
  eval_value_slots vars value_slots = Some vs ->
  closure_in_set statics v = Some (slot, function_decls, vs) ->
  type_satisfied statics vars v
    (closure_of_set_type slot function_decls (map fst value_slots) tys).
Proof.
  intros Hsat Htys Hvs Hclosure.
  apply (proj2 (exactly_this_closure_satisfied statics vars v slot _ _ _ _)).
  exists function_decls, vs. split; [ exact Hclosure | ].
  split; [ | split ].
  - intros name. unfold domain, map_values. rewrite map_map. reflexivity.
  - intros name. unfold domain. do 2 f_equal.
    transitivity (map fst value_slots).
    + apply map_fst_combine_same.
      rewrite <- (Forall2_length Htys). apply length_map.
    + symmetry. exact (eval_value_slots_names vars value_slots vs Hvs).
  - apply closures_entry_lookup_iff. split; [ | split ].
    + intros g function_type code Htype Hcode.
      rewrite lookup_map_values, Hcode in Htype. injection Htype as <-.
      reflexivity.
    + intros g ty closure Hty. discriminate Hty.
    + exact (value_slot_typed tenv statics vars Hsat value_slots tys vs Htys
               Hvs).
Qed.

Theorem closure_of_set_typed : closure_of_set_typed_statement.
Proof.
  intros tenv statics vars slot function_decls value_slots tys v Hsat Htys
    Hnamed.
  cbn [eval_named] in Hnamed.
  destruct (lookup slot function_decls) as [code | ] eqn:Hslot;
    [ | discriminate ].
  simpl in Hnamed.
  destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
    [ | discriminate ].
  injection Hnamed as <-.
  apply (closure_of_set_type_satisfied tenv statics vars slot function_decls
           value_slots tys vs _ Hsat Htys Hvs).
  simpl. rewrite Hslot. reflexivity.
Qed.

Theorem set_of_closures_typed : set_of_closures_typed_statement.
Proof.
  intros tenv statics vars code value_slots tys v Hsat Htys Hnamed.
  cbn [eval_named] in Hnamed.
  destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hvs;
    [ | discriminate ].
  injection Hnamed as <-.
  exact (closure_of_set_type_satisfied tenv statics vars
           unnamed_function_slot [(unnamed_function_slot, code)] value_slots
           tys vs (Value_closure code vs) Hsat Htys Hvs eq_refl).
Qed.

Theorem closure_of_set_type_free_variables
  : closure_of_set_type_free_variables_statement.
Proof.
  intros slot function_decls slots tys y.
  unfold closure_of_set_type, exactly_this_closure, create_closures. cbn.
  rewrite ?app_nil_r. intros Hin. apply in_flat_map in Hin.
  destruct Hin as [[w ty] [Hbinding Hfree]].
  exists ty. split; [ exact (in_combine_r _ _ _ _ Hbinding) | exact Hfree ].
Qed.

Theorem closure_of_set_type_rename : closure_of_set_type_rename_statement.
Proof.
  intros r slot function_decls slots tys. unfold closure_of_set_type.
  rewrite (proj1 (rename_type_of_closure r slot _ _ _ _)). f_equal.
  revert tys.
  induction slots as [ | w slots IH]; intros [ | ty tys]; simpl;
    try reflexivity.
  f_equal. apply IH.
Qed.

Lemma closure_type_in_scope (tenv : Typing_env.t) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (tys : list flambda_type) :
  Forall2 (argument_type tenv) (map snd value_slots) tys ->
  ∀ y : variable,
    y ∈ free_variables
          (closure_of_set_type slot function_decls (map fst value_slots) tys) ->
    in_scope_in_terms tenv y.
Proof.
  intros Htys y Hy.
  destruct (closure_of_set_type_free_variables slot function_decls
              (map fst value_slots) tys y Hy)
    as [ty [Hin Hfree]].
  clear Hy. induction Htys as [ | arg ty' args tys Hty Htys IH];
    [ contradiction | ].
  destruct Hin as [<- | Hin];
    [ exact (argument_type_in_scope tenv arg ty' y Hty Hfree)
    | exact (IH Hin) ].
Qed.

Theorem let_closure_of_set : let_closure_of_set_statement.
Proof.
  intros tenv x declared slot function_decls value_slots tys body body' Hwf
    Hfresh Htys closure Hbody.
  apply (let_typed_refines_under_pure tenv x declared closure body body'
           (closure_of_set_type slot function_decls (map fst value_slots)
              tys)
           eq_refl Hwf Hfresh).
  - exact (closure_type_in_scope tenv slot function_decls value_slots tys
             Htys).
  - intros statics vars v Hsat Hnamed.
    exact (closure_of_set_typed tenv statics vars slot function_decls
             value_slots tys v
             (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
             Htys Hnamed).
  - exact Hbody.
Qed.

Theorem let_set_of_closures : let_set_of_closures_statement.
Proof.
  intros tenv x declared code value_slots tys body body' Hwf Hfresh Htys
    closure Hbody.
  apply (let_typed_refines_under_pure tenv x declared closure body body'
           (set_of_closures_type code (map fst value_slots) tys) eq_refl Hwf
           Hfresh).
  - exact (closure_type_in_scope tenv unnamed_function_slot
             [(unnamed_function_slot, code)] value_slots tys Htys).
  - intros statics vars v Hsat Hnamed.
    exact (set_of_closures_typed tenv statics vars code value_slots tys v
             (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
             Htys Hnamed).
  - exact Hbody.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The call is what the function answers, computed. *)
Local Ltac call_computed :=
  match goal with
  | |- expr_refines_under ?tenv (Apply ?ap) _ =>
    pose proof (simplify_apply_refines_under tenv ap) as Hcall;
    match type of Hcall with
    | expr_refines_under _ _ ?answer =>
      let computed := eval vm_compute in answer in
      replace answer with computed in Hcall by (vm_compute; reflexivity)
    end;
    exact Hcall
  end.

(* The first [Let] of the three examples: [g] is given the type of the
   closure. *)
Local Ltac example_closure_let :=
  apply (let_set_of_closures call_env call_g Value call_code
           [(call_slot, Var call_x)] [alias_type_of Value (Var call_x)]);
  [ vm_compute; reflexivity | vm_compute; reflexivity
  | apply Forall2_cons; [ argument_computed | apply Forall2_nil ] | ].

Theorem example_local_function : example_local_function_statement.
Proof.
  intros k exn. unfold call_closure. example_closure_let.
  unfold call_of. call_computed.
Qed.

(* The variable [h] is made equal to [g]. *)
Local Ltac example_alias_let :=
  eapply (let_simple_refines_under _ call_h Value (Var call_g) _ _ Value);
  [ vm_compute; reflexivity | vm_compute; reflexivity
  | intros ? [= <-]; vm_compute; reflexivity
  | eexists; vm_compute; reflexivity | vm_compute; reflexivity | ].

Theorem example_closure_from_pair : example_closure_from_pair_statement.
Proof.
  intros k exn pair load. unfold call_closure. example_closure_let.
  apply (let_make_block _ call_p Value 0 [Var call_g; Var call_x]
           [alias_type_of Value (Var call_g);
            alias_type_of Value (Var call_x)]);
    [ vm_compute; reflexivity | vm_compute; reflexivity
    | repeat apply Forall2_cons;
      [ argument_computed | argument_computed | apply Forall2_nil ]
    | reflexivity | ].
  eapply expr_refines_under_trans.
  - (* The load is the [Simple] that the prover answers. *)
    match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_block_load tenv call_h Value
                    (Block_access_values (Some 0) (Some 2) Any_value)
                    Immutable 0 (Var call_p)
                    (alias_type_of Value (Var call_p)) body) as Hload
    end.
    cbv zeta in Hload.
    match type of Hload with
    | context [Provers.meet_block_field_simple ?tenv ?mode ?kind ?ty ?i] =>
      let answer :=
        eval vm_compute in
          (Provers.meet_block_field_simple tenv mode kind ty i)
      in
      replace (Provers.meet_block_field_simple tenv mode kind ty i)
        with answer in Hload by (vm_compute; reflexivity)
    end.
    cbv beta iota in Hload. apply Hload; [ discriminate | ].
    argument_computed.
  - example_alias_let. unfold call_of. call_computed.
Qed.

Theorem example_function_argument : example_function_argument_statement.
Proof.
  intros k exn. unfold call_closure. example_closure_let.
  example_alias_let. unfold call_of. call_computed.
Qed.
