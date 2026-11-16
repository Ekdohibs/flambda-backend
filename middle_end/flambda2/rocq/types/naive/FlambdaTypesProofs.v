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

(* Proofs of the statements of FlambdaTypesSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaApplySpec
  FlambdaApplyProofs.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec.
Import ListNotations.

(* ======================================================================== *)
(* Refinement relative to a typing environment                              *)
(* ======================================================================== *)

Lemma env_satisfies_empty (vars : env) : env_satisfies [] vars.
Proof. intros x t []. Qed.

Theorem expr_refines_weaken : expr_refines_weaken_statement.
Proof.
  intros tenv source target Hrefines vars conts statics hp rd _. apply Hrefines.
Qed.

Theorem expr_refines_in_empty : expr_refines_in_empty_statement.
Proof.
  intros source target. split; [ | apply expr_refines_weaken ].
  intros Hrefines vars conts statics hp rd. apply Hrefines, env_satisfies_empty.
Qed.

Lemma expr_equivalent_in_refines (tenv : typing_env) (e e' : expr) :
  expr_equivalent_in tenv e e' -> expr_refines_in tenv e e'.
Proof.
  intros Hequiv vars conts statics hp rd Hsat. apply equivalent_refines, Hequiv,
    Hsat.
Qed.

Lemma expr_refines_in_trans (tenv : typing_env) (e e' e'' : expr) :
  expr_refines_in tenv e e' -> expr_refines_in tenv e' e'' ->
  expr_refines_in tenv e e''.
Proof.
  intros H H' vars conts statics hp rd Hsat.
  exact (refines_trans _ _ _ (H vars conts statics hp rd_ Hsat)
    (H' vars conts statics hp rd_ Hsat)).
Qed.

Theorem bottom : bottom_statement.
Proof.
  intros tenv x e e' Hbottom vars conts statics hp rd Hsat.
  destruct (Hsat x Bottom Hbottom) as (v & _ & []).
Qed.

(* ======================================================================== *)
(* Following aliases                                                        *)
(* ======================================================================== *)

Lemma find_alias_in (tenv : typing_env) (x : variable) (s : simple) :
  find_alias tenv x = Some s -> (x, Equals s) ∈ tenv.
Proof.
  induction tenv as [ | [y t] tenv IH]; simpl; [ discriminate | ].
  destruct t as [ | | s' | code slots];
    try (intros Hfind; right; exact (IH Hfind)).
  destruct (Pos.eqb_spec x y) as [-> | _].
  - intros [= ->]. left. reflexivity.
  - intros Hfind. right. exact (IH Hfind).
Qed.

Theorem canonical_simple_sound : canonical_simple_sound_statement.
Proof.
  intros tenv vars s Hsat.
  destruct s as [x | sym | c]; simpl; try reflexivity.
  destruct (find_alias tenv x) as [alias | ] eqn:Hfind; [ | reflexivity ].
  destruct (Hsat x (Equals alias) (find_alias_in _ _ _ Hfind))
    as (v & Hlookup & Halias).
  simpl in Halias. rewrite Halias, Hlookup. reflexivity.
Qed.

Lemma canonical_simples_sound (tenv : typing_env) (vars : env)
    (args : list simple) :
  env_satisfies tenv vars ->
  eval_simples vars (map (canonical_simple tenv) args)
  = eval_simples vars args.
Proof.
  intros Hsat. induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite (canonical_simple_sound tenv vars s Hsat), IH. reflexivity.
Qed.

Theorem apply_cont_canonical : apply_cont_canonical_statement.
Proof.
  intros tenv k args vars conts statics hp rd Hsat.
  apply step_eq_equivalent. rewrite !step_apply_cont. simpl.
  rewrite (canonical_simples_sound tenv vars args Hsat). reflexivity.
Qed.

Theorem let_simple_canonical : let_simple_canonical_statement.
Proof.
  intros tenv x kind s body vars conts statics hp rd Hsat.
  apply let_equivalent_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | | intros v _ _;
     apply equivalent_refl ].
  exact (canonical_simple_sound tenv vars s Hsat).
Qed.

Theorem switch_known : switch_known_statement.
Proof.
  intros tenv scrutinee i arms ac Hknown Hfind vars conts statics hp rd Hsat.
  apply step_equivalent. rewrite step_switch.
  rewrite <- (canonical_simple_sound tenv vars scrutinee Hsat), Hknown.
  simpl. rewrite Hfind. reflexivity.
Qed.

(* ======================================================================== *)
(* Going under binders                                                      *)
(* ======================================================================== *)

(* Binding a variable that does not occur in the typing environment keeps
   it satisfied, and adds whatever type the value bound has. *)
Lemma eval_value_slots_extend (vars : env) (x : variable) (v : value)
    (slots : list (value_slot * simple)) :
  ~ Exists (fun slot => var_free_in_simple x (snd slot)) slots ->
  eval_value_slots ((x, v) :: vars) slots = eval_value_slots vars slots.
Proof.
  induction slots as [ | [slot s] slots IH]; intros Hnotfree; simpl;
    [ reflexivity | ].
  rewrite eval_simple_extend, IH; [ reflexivity | | ];
    intros Hfree; apply Hnotfree;
    [ apply Exists_cons_tl | apply Exists_cons_hd ]; exact Hfree.
Qed.

Lemma env_satisfies_extend (tenv : typing_env) (vars : env) (x : variable)
    (v : value) (t : flambda_type) :
  env_satisfies tenv vars ->
  ~ var_in_typing_env x tenv ->
  type_satisfied ((x, v) :: vars) v t ->
  env_satisfies ((x, t) :: tenv) ((x, v) :: vars).
Proof.
  intros Hsat Hfresh Ht y t' [[= <- <-] | Hin].
  - exists v. split; [ simpl; rewrite Pos.eqb_refl; reflexivity | exact Ht ].
  - destruct (Hsat y t' Hin) as (w & Hlookup & Hw).
    assert (Hne : y <> x).
    { intros ->. apply Hfresh. exists x, t'. auto. }
    exists w. split.
    + simpl. destruct (Pos.eqb_spec y x); [ contradiction | exact Hlookup ].
    + assert (Hnotfree : ~ var_free_in_type x t').
      { intros Hfree. apply Hfresh. exists y, t'. auto. }
      destruct t' as [ | | s | code slots]; simpl in *; try exact Hw.
      * rewrite eval_simple_extend; assumption.
      * destruct Hw as (slot_values & Heval & ->). exists slot_values.
        rewrite eval_value_slots_extend; auto.
Qed.

Theorem let_refines_in : let_refines_in_statement.
Proof.
  intros tenv x kind d body body' Hfresh Hbody vars conts statics hp rd Hsat.
  apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
  apply Hbody, env_satisfies_extend; [ exact Hsat | exact Hfresh | exact I ].
Qed.

Theorem let_simple_refines_in : let_simple_refines_in_statement.
Proof.
  intros tenv x kind s body body' Hfresh Hnotin Hbody vars conts statics hp rd
    Hsat.
  apply let_refines_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | reflexivity | intros
     v Hs _ ]. simpl in Hs.
  apply Hbody, env_satisfies_extend; [ exact Hsat | exact Hfresh | ].
  destruct s as [y | sym | c]; simpl in *; try exact Hs.
  destruct (Pos.eqb_spec y x) as [-> | _]; [ | exact Hs ].
  exfalso. apply Hnotin. reflexivity.
Qed.

Theorem let_cont_body_refines_in : let_cont_body_refines_in_statement.
Proof.
  intros tenv k h body body' Hbody vars conts statics hp rd Hsat.
  apply let_cont_non_recursive_refines_step, Hbody, Hsat.
Qed.

Theorem let_cont_recursive_body_refines_in :
  let_cont_recursive_body_refines_in_statement.
Proof.
  intros tenv handlers body body' Hbody vars conts statics hp rd Hsat.
  apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
  apply Hbody, Hsat.
Qed.

(* Binding parameters that do not occur in the typing environment keeps it
   satisfied. *)
Lemma env_satisfies_bind_params (tenv : typing_env) (params : list variable) :
  ∀ (vs : list value) (vars handler_vars : env),
    env_satisfies tenv vars ->
    (∀ param : variable,
      param ∈ params -> ~ var_in_typing_env param tenv) ->
    bind_params params vs vars = Some handler_vars ->
    env_satisfies tenv handler_vars.
Proof.
  induction params as [ | param params IH];
    intros [ | v vs] vars handler_vars Hsat Hfresh; simpl; try discriminate.
  - intros [= <-]. exact Hsat.
  - apply IH; [ | intros p Hin; apply Hfresh; right; exact Hin ].
    intros x t Hin.
    apply (env_satisfies_extend tenv vars param v Unknown Hsat);
      [ apply Hfresh; left; reflexivity | exact I | right; exact Hin ].
Qed.

Lemma env_satisfies_params (tenv : typing_env) (params : list variable)
    (vs : list value) (vars handler_vars : env) :
  env_satisfies tenv vars ->
  (∀ param : variable, param ∈ params -> ~ var_in_typing_env param tenv) ->
  bind_params params vs vars = Some handler_vars ->
  env_satisfies (add_definitions_of_params tenv params) handler_vars.
Proof.
  intros Hsat Hfresh Hbind x t Hin.
  apply in_app_or in Hin. destruct Hin as [Hin | Hin].
  - apply in_map_iff in Hin. destruct Hin as (param & [= <- <-] & Hin).
    destruct (bind_params_bound params vs vars handler_vars param Hbind)
      as [v Hlookup]; [ left; exact Hin | ].
    exists v. split; [ exact Hlookup | exact I ].
  - exact (env_satisfies_bind_params tenv params vs vars handler_vars Hsat
      Hfresh Hbind x t Hin).
Qed.

(* The handler refines where it runs: under the variables at the
   [Let_cont], which satisfy the typing environment, with its parameters
   bound. *)
Theorem let_cont_handler_refines_in : let_cont_handler_refines_in_statement.
Proof.
  intros tenv k params metadata handler handler' body Hfresh Hhandler vars
    conts statics hp rd Hsat.
  apply let_cont_handler_refines_at.
  intros vs handler_vars conts' statics' hp' rd' _ Hbind. apply Hhandler.
  exact (env_satisfies_params tenv params vs vars handler_vars Hsat Hfresh
    Hbind).
Qed.

Theorem let_cont_recursive_handler_refines_in :
  let_cont_recursive_handler_refines_in_statement.
Proof.
  intros tenv before k params metadata handler handler' after body Hfresh
    Hbinds Hhandler vars conts statics hp rd Hsat.
  apply let_cont_recursive_handler_refines_at; [ exact Hbinds | ].
  intros vs handler_vars conts' statics' hp' rd' _ Hbind. apply Hhandler.
  exact (env_satisfies_params tenv params vs vars handler_vars Hsat Hfresh
    Hbind).
Qed.

(* ======================================================================== *)
(* An example: propagating a constant                                       *)
(* ======================================================================== *)

(* Under the [Let] the variable is known to be the constant, so its use
   may be replaced; the [Let] is then unused. *)
Theorem let_const_apply_cont : let_const_apply_cont_statement.
Proof.
  intros x kind c k.
  assert (Hrewrite :
    expr_refines
      (Let x kind (Simple (Const c))
         (Apply_cont (Apply_cont_expr k [Var x] None)))
      (Let x kind (Simple (Const c))
         (Apply_cont (Apply_cont_expr k [Const c] None)))).
  { apply expr_refines_in_empty, let_simple_refines_in.
    - unfold var_in_typing_env. intros (y & t & [] & _).
    - simpl. intros [].
    - apply expr_equivalent_in_refines.
      pose proof (apply_cont_canonical [(x, Equals (Const c))] k [Var x])
        as Hcanonical.
      simpl in Hcanonical. rewrite Pos.eqb_refl in Hcanonical.
      exact Hcanonical. }
  assert (Hunused :
    expr_refines
      (Let x kind (Simple (Const c))
         (Apply_cont (Apply_cont_expr k [Const c] None)))
      (Apply_cont (Apply_cont_expr k [Const c] None))).
  { apply unused_let; [ reflexivity | ]. simpl.
    unfold var_free_in_simples. intros Hfree.
    inversion Hfree as [? ? Hhead | ? ? Hrest];
      [ exact Hhead | inversion Hrest ]. }
  intros vars conts statics hp rd.
  exact (refines_trans _ _ _ (Hrewrite vars conts statics hp rd_)
    (Hunused vars conts statics hp rd_)).
Qed.

Theorem let_simple_apply_cont : let_simple_apply_cont_statement.
Proof.
  intros x kind s k Hnotfree.
  assert (Hrewrite :
    expr_refines
      (Let x kind (Simple s) (Apply_cont (Apply_cont_expr k [Var x] None)))
      (Let x kind (Simple s) (Apply_cont (Apply_cont_expr k [s] None)))).
  { apply expr_refines_in_empty, let_simple_refines_in.
    - unfold var_in_typing_env. intros (y & t & [] & _).
    - exact Hnotfree.
    - apply expr_equivalent_in_refines.
      pose proof (apply_cont_canonical [(x, Equals s)] k [Var x])
        as Hcanonical.
      simpl in Hcanonical. rewrite Pos.eqb_refl in Hcanonical.
      exact Hcanonical. }
  assert (Hunused :
    expr_refines
      (Let x kind (Simple s) (Apply_cont (Apply_cont_expr k [s] None)))
      (Apply_cont (Apply_cont_expr k [s] None))).
  { apply unused_let; [ reflexivity | ]. simpl.
    unfold var_free_in_simples. intros Hfree.
    inversion Hfree as [? ? Hhead | ? ? Hrest];
      [ exact (Hnotfree Hhead) | inversion Hrest ]. }
  intros vars conts statics hp rd.
  exact (refines_trans _ _ _ (Hrewrite vars conts statics hp rd_)
    (Hunused vars conts statics hp rd_)).
Qed.

(* ======================================================================== *)
(* An example: inlining the identity function                               *)
(* ======================================================================== *)

Theorem inline_identity : inline_identity_statement.
Proof.
  intros called x my_closure ret exn metadata callee arg k k_exn args_arity
    return_arity Hcode_conts Hcall_conts Hx Hmy_closure Hfull Htupled.
  set (bindings :=
    [Bind_code called (identity_code x my_closure ret exn metadata)]).
  set (ap :=
    Apply_expr callee (Direct called) [arg] k k_exn [] args_arity
      return_arity).
  set (kind := hd Value (unarize (cm_params_arity metadata))).
  destruct (inline_renaming_exists ret exn k k_exn Hcode_conts Hcall_conts)
    as (r & Hvars & Hret & Hexn).
  (* the call inlined *)
  assert (Hinline :
    expr_refines
      (Let_static bindings (Apply ap))
      (Let_static bindings
         (Let my_closure Value (Simple callee)
            (Let x kind (Simple arg)
               (Apply_cont (Apply_cont_expr k [Var x] None)))))).
  { pose proof (inline_let_code ap called
      (identity_code x my_closure ret exn metadata) r eq_refl) as Hinline.
    unfold inlined_body, identity_code in Hinline. simpl in Hinline.
    unfold rename_apply_cont in Hinline. simpl in Hinline.
    rewrite Hret, Hvars in Hinline. apply Hinline.
    unfold inlinable. simpl. repeat split.
    - intros y Hy. unfold var_free_in_simples in Hy.
      inversion Hy as [? ? Hhead | ? ? Hrest]; [ | inversion Hrest ].
      right. left. simpl in Hhead. symmetry. exact Hhead.
    - intros j [<- | []]. left. reflexivity.
    - exact Hvars.
    - exact Hret.
    - exact Hexn.
    - unfold var_free_in_simples. intros Hfree.
      inversion Hfree as [? ? Hhead | ? ? Hrest];
        [ exact (Hmy_closure Hhead) | inversion Hrest ].
    - intros param [<- | []]. unfold var_free_in_simples. intros Hfree.
      inversion Hfree as [? ? Hhead | ? ? Hrest];
        [ exact (Hx Hhead) | inversion Hrest ].
    - exact Hfull.
    - exact Htupled. }
  (* the parameter replaced by the argument, under the bindings *)
  assert (Hparam :
    expr_refines
      (Let_static bindings
         (Let my_closure Value (Simple callee)
            (Let x kind (Simple arg)
               (Apply_cont (Apply_cont_expr k [Var x] None)))))
      (Let_static bindings
         (Let my_closure Value (Simple callee)
            (Apply_cont (Apply_cont_expr k [arg] None))))).
  { exact (plug_refines
      (Let_static_body bindings
         (Let_body my_closure Value (Simple callee) Hole))
      _ _ (not_many _) (let_simple_apply_cont x kind arg k Hx)). }
  (* the first [Let] is unused *)
  assert (Hunused :
    expr_refines
      (Let_static bindings
         (Let my_closure Value (Simple callee)
            (Apply_cont (Apply_cont_expr k [arg] None))))
      (Let_static bindings (Apply_cont (Apply_cont_expr k [arg] None)))).
  { apply (plug_refines (Let_static_body bindings Hole));
      [ apply not_many | ].
    apply unused_let; [ reflexivity | ]. simpl.
    unfold var_free_in_simples. intros Hfree.
    inversion Hfree as [? ? Hhead | ? ? Hrest];
      [ exact (Hmy_closure Hhead) | inversion Hrest ]. }
  intros vars conts statics hp rd.
  exact (refines_trans _ _ _ (Hinline vars conts statics hp rd)
    (refines_trans _ _ _ (Hparam vars conts statics hp rd)
       (Hunused vars conts statics hp rd))).
Qed.
