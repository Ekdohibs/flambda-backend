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


(* Proofs of the statements of FlambdaUnboxVariantsSpec.v. Nothing here
   needs review.

   Giving a continuation one more parameter and every application one
   more argument is the simulation of FlambdaUnboxParamsProofs.v
   ([unbox_parameter_refines]), which asks two things: of the new
   argument, that it has a value of the kind of the parameter which
   the primitive gives of the parameter read, if it gives one; and of
   the two handlers, that the one without the [Let] refines the other
   when its new parameter holds such a value ([read_refines], by the
   lemmas of contexts, with no simulation of its own). Here are the
   first for the arguments of [extend_args_certain], that
   [first_read] finds a [Let] of which the second holds, and the
   pass. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaBlockLoadSpec
  FlambdaBlockLoadProofs FlambdaUnboxSpec FlambdaUnboxProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaUnusedContsSpec FlambdaUnusedContsProofs
  FlambdaCompareSpec FlambdaCompareProofs FlambdaUnboxParamsSpec
  FlambdaUnboxParamsProofs FlambdaUnboxVariantsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The argument for a parameter that is read under a test                   *)
(* ======================================================================== *)

Lemma dummy_of_kind (variant : bool) (p : unary_primitive)
    (kind : flambda_kind) :
  kind_of_const (dummy_of variant p kind) = kind.
Proof.
  destruct p, variant; simpl; try reflexivity;
    (destruct (kind_eqb kind (Naked_number Kind_naked_immediate)) eqn:Hkind;
     [ apply kind_eqb_eq in Hkind; subst kind | ]; reflexivity).
Qed.

(* Where the term shows that the primitive has no value, it has
   none. *)
Lemma no_value_on_sound (known : facts) (vars : env) (statics : static_env)
    (p : unary_primitive) (arg : simple) (v : value) :
  facts_hold known vars statics ->
  no_value_on known p arg = true ->
  eval_simple vars arg = Some v ->
  ∀ (statics' : static_env) (w : value),
    eval_prim statics' (Unary p) [v] = Some w -> False.
Proof.
  intros ([Hvars _] & _ & _) Hnone Harg statics' w.
  rewrite eval_prim_unary.
  destruct p, arg as [y | sym | c]; simpl in Hnone; try discriminate Hnone.
  - (* a load of a constant *)
    injection Harg as <-. simpl. destruct kind; discriminate.
  - (* [Get_tag] of a constant *)
    injection Harg as <-. discriminate.
  - (* the untagging of a block *)
    unfold is_bound in Hnone.
    destruct (lookup y (known_vars (facts_blocks known))) as [block | ]
      eqn:Hblock; [ | discriminate ].
    destruct (Hvars (y, block) (lookup_in _ _ _ Hblock)) as (vs & _ & Hy).
    simpl in Harg, Hy. rewrite Hy in Harg. injection Harg as <-.
    discriminate.
  - (* the untagging of a symbol *)
    injection Harg as <-. discriminate.
Qed.

(* The field of a block that was made: it has a value of the kind of
   values, which is what the load gives of the block if it gives
   anything. *)
Lemma made_field_gives (known : facts) (vars : env) (statics : static_env)
    (p : unary_primitive) (kind : flambda_kind) (arg projection : simple)
    (v : value) :
  facts_hold known vars statics ->
  made_field known p kind arg = Some projection ->
  eval_simple vars arg = Some v ->
  ∃ u : value,
    eval_simple vars projection = Some u
    ∧ kind_of_value u = kind
    ∧ gives p v u statics.
Proof.
  intros ([Hvars _] & _ & Hvalued & _) Hmade Harg.
  unfold made_field in Hmade.
  destruct p as [access mut field | | | | | | | | | | | | | | ];
    try discriminate Hmade.
  assert (Hfield :
    match arg with
    | Var y =>
      let* block := lookup y (known_vars (facts_blocks known)) in
      let* contents := field_loaded access field block in
      if kind_eqb kind Value then Some contents else None
    | _ => None
    end
    = Some projection)
    by (destruct mut; [ discriminate Hmade | exact Hmade .. ]).
  clear Hmade. destruct arg as [y | sym | c]; try discriminate Hfield.
  destruct (lookup y (known_vars (facts_blocks known))) as [block | ]
    eqn:Hblock; [ | discriminate ].
  simpl in Hfield.
  destruct (field_loaded access field block) as [contents | ] eqn:Hloaded;
    [ | discriminate ].
  simpl in Hfield.
  destruct (kind_eqb kind Value) eqn:Hkind; [ | discriminate ].
  injection Hfield as <-. apply kind_eqb_eq in Hkind. subst kind.
  pose proof (lookup_in _ _ _ Hblock) as Hin.
  destruct (Hvars (y, block) Hin) as (vs & Hfields & Hy).
  simpl in Hfields, Hy, Harg. rewrite Hy in Harg. injection Harg as <-.
  destruct (field_loaded_spec _ _ _ _ Hloaded)
    as (tag & size & -> & _ & _ & _ & Hnth).
  destruct (fields_hold_nth _ _ _ Hfields _ _ Hnth)
    as (loaded & Hnth_loaded & Hcontents).
  exists loaded. split; [ exact Hcontents | split ].
  - exact (proj1 (Forall_forall _ _)
      (Hvalued y block (fst block) vs Hin Hy) loaded
      (nth_error_In _ _ Hnth_loaded)).
  - intros statics' _ w. rewrite eval_prim_unary. simpl.
    destruct (known_is tag (fst block)
              && known_is size (Z.of_nat (length vs)) && (0 <=? field)%Z);
      [ | discriminate ].
    rewrite Hnth_loaded. simpl.
    destruct (kind_eqb (kind_of_value loaded) Value); [ | discriminate ].
    intros [= <-]. reflexivity.
Qed.

(* The number in a box that was made: it has a value of the kind of
   the number, which is what the unboxing gives of the box if it gives
   anything. *)
Lemma made_number_gives (known : facts) (vars : env) (statics : static_env)
    (p : unary_primitive) (kind : flambda_kind) (arg projection : simple)
    (v : value) :
  facts_hold known vars statics ->
  made_number known p kind arg = Some projection ->
  eval_simple vars arg = Some v ->
  ∃ u : value,
    eval_simple vars projection = Some u
    ∧ kind_of_value u = kind
    ∧ gives p v u statics.
Proof.
  intros (_ & Hnumbers & _ & Hkinded & _) Hmade Harg.
  unfold made_number in Hmade.
  destruct p as [ | | | | | | | | | | boxed | | | | ]; try discriminate Hmade;
    (destruct arg as [y | sym | c]; try discriminate Hmade).
  - (* The number in a box. *)
    destruct (unwrapped (facts_numbers known) (Boxed boxed) (Var y))
      as [contents | ] eqn:Hunwrapped; [ | discriminate ].
    simpl in Hmade.
    destruct (kind_eqb kind (unboxed_kind boxed)) eqn:Hkind;
      [ | discriminate ].
    injection Hmade as <-. apply kind_eqb_eq in Hkind. subst kind.
    destruct (unwrapped_var_sound _ _ _ _ _ _ Hnumbers Hunwrapped)
      as (number & box & Hcontents & Hy & c & -> & ->).
    assert (Hin :
      (y, (Boxed boxed, contents))
      ∈ known_wrapped_vars (facts_numbers known)).
    { simpl in Hunwrapped.
      destruct (lookup y (known_wrapped_vars (facts_numbers known)))
        as [[how' contents'] | ] eqn:Hlookup; [ | discriminate ].
      destruct (wrapping_eqb (Boxed boxed) how') eqn:Hhow; [ | discriminate ].
      apply wrapping_eqb_eq in Hhow. subst how'.
      injection Hunwrapped as ->. exact (lookup_in _ _ _ Hlookup). }
    pose proof (Hkinded y boxed contents c Hin Hy) as Hnumber.
    simpl in Harg. rewrite Hy in Harg. injection Harg as <-.
    exists (Value_const c). split; [ exact Hcontents | split ].
    + exact (is_number_of_kind_kind boxed c Hnumber).
    + intros statics' _ w. rewrite eval_prim_unary. simpl.
      unfold boxable_contents.
      destruct (boxable_number_of_const c); [ | discriminate ].
      simpl. rewrite Hnumber. intros [= <-]. reflexivity.
  - (* The number that was tagged: what is known of the variable says
       that it holds a tagged immediate. *)
    destruct (unwrapped (facts_numbers known) Tagged (Var y))
      as [contents | ] eqn:Hunwrapped; [ | discriminate ].
    simpl in Hmade.
    destruct (kind_eqb kind (Naked_number Kind_naked_immediate)) eqn:Hkind;
      [ | discriminate ].
    injection Hmade as <-. apply kind_eqb_eq in Hkind. subst kind.
    destruct (unwrapped_var_sound _ _ _ _ _ _ Hnumbers Hunwrapped)
      as (number & tagged & Hcontents & Hy & i & -> & ->).
    simpl in Harg. rewrite Hy in Harg. injection Harg as <-.
    exists (Value_const (Naked_immediate i)).
    split; [ exact Hcontents | split; [ reflexivity | ] ].
    intros statics' _ w. rewrite eval_prim_unary. simpl.
    destruct (in_range _ i); [ | discriminate ].
    intros [= <-]. reflexivity.
Qed.

(* ======================================================================== *)
(* A read that an application cannot come to                                *)
(* ======================================================================== *)

(* The variables that are known to hold a number hold it. *)
Definition immediates_hold (consts : list (variable * Z)) (vars : env)
    : Prop :=
  ∀ (t : variable) (i : Z),
    lookup t consts = Some i ->
    lookup t vars = Some (Value_const (Naked_immediate i)).

Lemma lookup_forget_immediates (bound : list variable)
    (consts : list (variable * Z)) (t : variable) (i : Z) :
  lookup t (forget_immediates bound consts) = Some i ->
  lookup t consts = Some i ∧ among t bound = false.
Proof.
  unfold forget_immediates.
  induction consts as [ | [t0 i0] consts IH]; [ discriminate | ].
  cbn [filter fst lookup].
  destruct (among t0 bound) eqn:Hbound; cbn [negb lookup].
  - intros Hlookup. destruct (IH Hlookup) as [Hfound Hfree].
    split; [ | exact Hfree ].
    destruct (Pos.eqb_spec t t0) as [-> | _]; [ congruence | exact Hfound ].
  - destruct (Pos.eqb_spec t t0) as [-> | _]; [ | exact IH ].
    intros [= <-]. auto.
Qed.

Lemma immediates_hold_under_let (consts : list (variable * Z)) (vars : env)
    (y : variable) (v : value) :
  immediates_hold consts vars ->
  immediates_hold (forget_immediates [y] consts) ((y, v) :: vars).
Proof.
  intros Hhold t i Ht.
  destruct (lookup_forget_immediates [y] consts t i Ht) as [Hfound Hfree].
  rewrite among_cons in Hfree. apply orb_false_iff in Hfree as [Hne _].
  apply Pos.eqb_neq in Hne. rewrite lookup_other by exact Hne.
  exact (Hhold t i Hfound).
Qed.

Lemma immediates_hold_bind_params (consts : list (variable * Z))
    (params : list variable) (vs : list value) (vars handler_vars : env) :
  immediates_hold consts vars ->
  bind_params params vs vars = Some handler_vars ->
  immediates_hold (forget_immediates params consts) handler_vars.
Proof.
  intros Hhold Hbind t i Ht.
  destruct (lookup_forget_immediates params consts t i Ht) as [Hfound Hfree].
  rewrite (bind_params_other params t vs vars handler_vars Hfree Hbind).
  exact (Hhold t i Hfound).
Qed.

(* Taking the arms that the known numbers lead to gives a term that
   refines the term, and that the term refines. *)
Fixpoint take_arms_refines (e : expr) :
  ∀ (consts : list (variable * Z)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    immediates_hold consts vars ->
    refines (State e vars conts statics hp)
      (State (take_arms consts e) vars conts statics hp).
Proof.
  destruct e
    as [ y kind d body | bindings body | k [handler_params metadata handler]
           body
       | handlers body | ap | ac | scrutinee arms | ];
    intros consts vars conts statics hp Hhold; cbn [take_arms];
    try apply refines_refl.
  - apply let_refines_step; [ reflexivity | ].
    intros v _ _. apply take_arms_refines, immediates_hold_under_let, Hhold.
  - apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    apply take_arms_refines, Hhold.
  - apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler handler_params metadata
               (take_arms (forget_immediates handler_params consts) handler))
            body)
         vars conts statics hp)).
    + apply let_cont_handler_refines_at.
      intros vs handler_vars conts0 statics0 hp' _ Hbind.
      apply take_arms_refines.
      exact (immediates_hold_bind_params consts handler_params vs vars
        handler_vars Hhold Hbind).
    + apply let_cont_non_recursive_refines_step.
      apply take_arms_refines, Hhold.
  - apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
    apply take_arms_refines, Hhold.
  - destruct scrutinee as [t | sym | c]; try apply refines_refl.
    destruct (lookup t consts) as [i | ] eqn:Ht; [ | apply refines_refl ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | apply refines_refl ].
    apply equivalent_refines, step_equivalent. rewrite step_switch. simpl.
    rewrite (Hhold t i Ht). simpl. rewrite Harm. reflexivity.
Qed.

Fixpoint take_arms_refined (e : expr) :
  ∀ (consts : list (variable * Z)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    immediates_hold consts vars ->
    refines (State (take_arms consts e) vars conts statics hp)
      (State e vars conts statics hp).
Proof.
  destruct e
    as [ y kind d body | bindings body | k [handler_params metadata handler]
           body
       | handlers body | ap | ac | scrutinee arms | ];
    intros consts vars conts statics hp Hhold; cbn [take_arms];
    try apply refines_refl.
  - apply let_refines_step; [ reflexivity | ].
    intros v _ _. apply take_arms_refined, immediates_hold_under_let, Hhold.
  - apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    apply take_arms_refined, Hhold.
  - apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler handler_params metadata
               (take_arms (forget_immediates handler_params consts) handler))
            body)
         vars conts statics hp)).
    + apply let_cont_non_recursive_refines_step.
      apply take_arms_refined, Hhold.
    + apply let_cont_handler_refines_at.
      intros vs handler_vars conts0 statics0 hp' _ Hbind.
      apply take_arms_refined.
      exact (immediates_hold_bind_params consts handler_params vs vars
        handler_vars Hhold Hbind).
  - apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
    apply take_arms_refined, Hhold.
  - destruct scrutinee as [t | sym | c]; try apply refines_refl.
    destruct (lookup t consts) as [i | ] eqn:Ht; [ | apply refines_refl ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | apply refines_refl ].
    apply equivalent_refines, equivalent_sym, step_equivalent.
    rewrite step_switch. simpl.
    rewrite (Hhold t i Ht). simpl. rewrite Harm. reflexivity.
Qed.

(* Where an application does not come to the read, the handler without
   the read refines the handler, under the numbers that are known. *)
Lemma never_reads_refines (consts : list (variable * Z))
    (handler without : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) :
  never_reads consts handler without = true ->
  immediates_hold consts vars ->
  refines (State handler vars conts statics hp)
    (State without vars conts statics hp).
Proof.
  intros Hsame Hhold. unfold never_reads in Hsame.
  eapply refines_trans;
    [ exact (take_arms_refines handler consts vars conts statics hp Hhold) | ].
  eapply refines_trans;
    [ apply equivalent_refines;
      exact (remove_unused_conts_equivalent (take_arms consts handler) vars
        conts statics hp)
    | ].
  eapply refines_trans;
    [ apply equivalent_refines;
      exact (expr_eqb_sound _ _ Hsame vars conts statics hp)
    | ].
  eapply refines_trans;
    [ apply equivalent_refines, equivalent_sym;
      exact (remove_unused_conts_equivalent (take_arms consts without) vars
        conts statics hp)
    | ].
  exact (take_arms_refined without consts vars conts statics hp Hhold).
Qed.

(* A parameter that is known to be given a number is given that
   constant. *)
Lemma known_immediates_spec (params : list variable) :
  ∀ (args : list simple) (t : variable) (i : Z),
    lookup t (known_immediates params args) = Some i ->
    ∃ j : nat,
      nth_error params j = Some t
      ∧ nth_error args j = Some (Const (Naked_immediate i)).
Proof.
  induction params as [ | param params IH]; intros [ | arg args] t i;
    try discriminate.
  assert (Hrest :
    lookup t (known_immediates params args) = Some i ->
    ∃ j : nat,
      nth_error (param :: params) j = Some t
      ∧ nth_error (arg :: args) j = Some (Const (Naked_immediate i))).
  { intros Hlookup. destruct (IH args t i Hlookup) as (j & Hparam & Harg).
    exists (S j). auto. }
  destruct arg as [x | sym | c]; try exact Hrest.
  destruct c; try exact Hrest.
  cbn [known_immediates lookup].
  destruct (Pos.eqb_spec t param) as [-> | _]; [ | exact Hrest ].
  intros [= <-]. exists 0. auto.
Qed.

(* A variable that is not free may be bound to anything. *)
Lemma unread_refines (f : variable) (u : value) (e : expr) (bound : env)
    (conts : cont_env) (statics : static_env) (hp : heap) :
  var_free_b f e = false ->
  refines (State e bound conts statics hp)
    (State e ((f, u) :: bound) conts statics hp).
Proof.
  intros Hfree. apply equivalent_refines, coincide_equivalent.
  - intros y Hy. apply lookup_other. intros ->.
    apply var_free_b_sound in Hy. congruence.
  - intros k _. apply conts_agree_same. reflexivity.
Qed.

(* What is known of some of the variables is known. *)
Lemma lookup_only_of (tests : list variable) (t : variable) (c : Z) :
  ∀ consts : list (variable * Z),
    lookup t (only_of tests consts) = Some c -> lookup t consts = Some c.
Proof.
  unfold only_of.
  induction consts as [ | [t' c'] consts IH]; [ discriminate | ].
  cbn [filter fst].
  destruct (among t' tests) eqn:Hamong; cbn [lookup].
  - destruct (Pos.eqb t t'); [ intros H; exact H | exact IH ].
  - destruct (Pos.eqb_spec t t') as [-> | _]; [ | exact IH ].
    intros Hlookup. exfalso. clear IH. revert Hlookup.
    induction consts as [ | [t'' c''] consts IH]; [ discriminate | ].
    cbn [filter fst].
    destruct (among t'' tests) eqn:Hamong''; [ | exact IH ].
    cbn [lookup].
    destruct (Pos.eqb_spec t' t'') as [-> | _]; [ congruence | exact IH ].
Qed.

(* What is asked of the value that [extend_args_certain] gives for the
   new parameter: it is what the primitive gives of the parameter
   read; or the application does not come to the read, by the numbers
   that it gives. *)
Definition certain_fits (params : list variable) (handler without : expr)
    (i : nat) (p : unary_primitive) (statics : static_env)
    (vs : list value) (u : value) : Prop :=
  read_fits i p statics vs u
  ∨ ∃ consts : list (variable * Z),
      never_reads consts handler without = true
      ∧ ∀ (t : variable) (c : Z),
          lookup t consts = Some c ->
          ∃ j : nat,
            nth_error params j = Some t
            ∧ nth_error vs j = Some (Value_const (Naked_immediate c)).

(* The argument that [extend_args_certain] gives has a value of the
   kind given, which is as asked. *)
Lemma extend_args_certain_gives (variant : bool)
    (tests params : list variable)
    (handler without : expr) (i : nat) (p : unary_primitive)
    (kind : flambda_kind) (known : facts) (vars : env)
    (statics : static_env) (args args' : list simple) (vs : list value)
    (v : value) :
  facts_hold known vars statics ->
  extend_args_certain variant tests params handler without known i p kind
    args
  = Some args' ->
  eval_simples vars args = Some vs ->
  nth_error vs i = Some v ->
  ∃ u : value,
    eval_simples vars args' = Some (vs ++ [u])
    ∧ kind_of_value u = kind
    ∧ certain_fits params handler without i p statics vs u.
Proof.
  intros Hfacts Hextend Hvs Hv. unfold extend_args_certain in Hextend.
  destruct (nth_error args i) as [arg | ] eqn:Hnth; [ | discriminate ].
  simpl in Hextend.
  destruct (eval_simples_nth vars args vs i arg Hvs Hnth)
    as (v' & Hv' & Harg).
  rewrite Hv in Hv'. injection Hv' as <-.
  destruct
    (never_reads (only_of tests (known_immediates params args)) handler
       without)
    eqn:Hnever.
  { injection Hextend as <-.
    exists (Value_const (dummy_of variant p kind)). split; [ | split ].
    - exact (eval_simples_snoc vars args (Const (dummy_of variant p kind))
        (Value_const (dummy_of variant p kind)) vs Hvs eq_refl).
    - apply dummy_of_kind.
    - right. exists (only_of tests (known_immediates params args)).
      split; [ exact Hnever | ].
      intros t c Ht. apply lookup_only_of in Ht.
      destruct (known_immediates_spec params args t c Ht)
        as (j & Hparam & Hgiven).
      exists j. split; [ exact Hparam | ].
      destruct (eval_simples_nth vars args vs j _ Hvs Hgiven)
        as (w & Hw & Heval).
      simpl in Heval. injection Heval as <-. exact Hw. }
  destruct (certain_projection known p kind arg) as [projection | ]
    eqn:Hcertain.
  - injection Hextend as <-.
    assert (Hprojection :
      ∃ u : value,
        eval_simple vars projection = Some u
        ∧ kind_of_value u = kind
        ∧ gives p v u statics).
    { unfold certain_projection in Hcertain.
      assert (Hmade :
        match made_field known p kind arg with
        | Some contents => Some contents
        | None => made_number known p kind arg
        end
        = Some projection ->
        ∃ u : value,
          eval_simple vars projection = Some u
          ∧ kind_of_value u = kind
          ∧ gives p v u statics).
      { destruct (made_field known p kind arg) as [contents | ] eqn:Hfield.
        - intros [= <-].
          exact (made_field_gives known vars statics p kind arg contents v
            Hfacts Hfield Harg).
        - intros Hnumber.
          exact (made_number_gives known vars statics p kind arg projection
            v Hfacts Hnumber Harg). }
      destruct (projection_of known p arg) as [[y | sym | c] | ]
        eqn:Hknown;
        try exact (Hmade Hcertain).
      destruct (kind_eqb (kind_of_const c) kind) eqn:Hkind;
        [ | discriminate ].
      injection Hcertain as <-. exists (Value_const c).
      split; [ reflexivity | split; [ apply kind_eqb_eq, Hkind | ] ].
      intros statics' Hmore w Hprim.
      pose proof (projection_of_sound known vars statics' p arg (Const c) v
        w (facts_hold_extend known vars statics statics' Hfacts Hmore)
        Hknown Harg Hprim) as Hw.
      simpl in Hw. congruence. }
    destruct Hprojection as (u & Hu & Hkind & Hgives).
    exists u. split; [ | split; [ exact Hkind | ] ].
    + exact (eval_simples_snoc vars args projection u vs Hvs Hu).
    + left. exists v. auto.
  - destruct (no_value_on known p arg) eqn:Hnone; [ | discriminate ].
    injection Hextend as <-.
    exists (Value_const (dummy_of variant p kind)). split; [ | split ].
    + exact (eval_simples_snoc vars args (Const (dummy_of variant p kind))
        (Value_const (dummy_of variant p kind)) vs Hvs eq_refl).
    + apply dummy_of_kind.
    + left. exists v. split; [ exact Hv | ].
      intros statics' _ w Hprim. exfalso.
      exact (no_value_on_sound known vars statics p arg v Hfacts Hnone Harg
        statics' w Hprim).
Qed.

(* ======================================================================== *)
(* The first read of a parameter                                            *)
(* ======================================================================== *)

Lemma param_read_spec (wanted : unary_primitive -> bool)
    (params bound : list variable) (f : variable) (d : named)
    (p : unary_primitive) (i : nat) :
  param_read wanted params bound f d = Some (p, i) ->
  ∃ x : variable,
    d = Prim (Unary p) [Var x]
    ∧ position x params = Some i
    ∧ among x bound = false
    ∧ among f bound = false.
Proof.
  unfold param_read.
  destruct d as [s | pr args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try discriminate.
  destruct pr as [u | b | v | hp | rp]; try discriminate.
  destruct args as [ | [x | sym | c] [ | arg args]]; try discriminate.
  destruct
    (wanted u && reads_a_box u && negb (among x bound)
     && negb (among f bound))
    eqn:Hchecks; [ | discriminate ].
  apply andb_true_iff in Hchecks as [Hchecks Hf].
  apply andb_true_iff in Hchecks as [_ Hx].
  apply negb_true_iff in Hx, Hf.
  destruct (position x params) as [j | ] eqn:Hposition; [ | discriminate ].
  simpl. intros [= <- <-]. exists x. auto.
Qed.

(* What [load_read] gives is the variable read. *)
Lemma load_read_spec (p : unary_primitive) (x x0 : variable) (field : Z) :
  load_read (Prim (Unary p) [Var x]) = Some (x0, field) -> x0 = x.
Proof. destruct p; simpl; try discriminate. intros [= <- _]. reflexivity. Qed.

Lemma position_single (x y : variable) (i : nat) :
  position x [y] = Some i -> x = y.
Proof.
  simpl. destruct (Pos.eqb_spec x y) as [-> | _]; [ reflexivity | ].
  discriminate.
Qed.

(* What [first_read] finds is a [Let] of the expression that reads a
   parameter which nothing on the way binds, and the expression without
   it. *)
Fixpoint first_read_removes (e : expr) :
  ∀ (wanted : unary_primitive -> bool) (params bound : list variable)
    (r : read),
    first_read wanted params bound e = Some r ->
    ∃ x : variable,
      position x params = Some (read_position r)
      ∧ among x bound = false
      ∧ among (read_variable r) bound = false
      ∧ removes (read_variable r) (read_kind r) (read_primitive r) x e
          (read_without r).
Proof.
  destruct e
    as [ y kind d body | bindings body | k [handler_params metadata handler]
           body
       | handlers body | ap | ac | scrutinee arms | ];
    intros wanted params bound r; simpl; try discriminate.
  - destruct (param_read wanted params bound y d) as [[p i] | ] eqn:Hread.
    + destruct (param_read_spec wanted params bound y d p i Hread)
        as (x & -> & Hposition & Hx & Hy).
      destruct (load_read (Prim (Unary p) [Var x])) as [[x0 field] | ]
        eqn:Hload; simpl.
      * apply load_read_spec in Hload. subst x0.
        destruct (first_read (loads_above field) [x] (y :: bound) body)
          as [r0 | ] eqn:Hbelow.
        -- intros [= <-]. simpl.
           destruct
             (first_read_removes body (loads_above field) [x] (y :: bound)
                r0 Hbelow)
             as (x' & Hposition' & Hx' & Hf' & Hremoves).
           apply position_single in Hposition'. subst x'.
           rewrite among_cons in Hx', Hf'.
           apply orb_false_iff in Hx' as [Hxy _].
           apply orb_false_iff in Hf' as [Hfy Hf].
           exists x. split; [ exact Hposition | split; [ exact Hx | ] ].
           split; [ exact Hf | ].
           apply Removes_under_let; assumption.
        -- intros [= <-]. simpl.
           exists x. split; [ exact Hposition | split; [ exact Hx | ] ].
           split; [ exact Hy | apply Removes_here ].
      * intros [= <-]. simpl.
        exists x. split; [ exact Hposition | split; [ exact Hx | ] ].
        split; [ exact Hy | apply Removes_here ].
    + destruct (first_read wanted params (y :: bound) body) as [r0 | ]
        eqn:Hbody; [ | discriminate ].
      intros [= <-]. simpl.
      destruct (first_read_removes body wanted params (y :: bound) r0 Hbody)
        as (x & Hposition & Hx & Hf & Hremoves).
      rewrite among_cons in Hx, Hf.
      apply orb_false_iff in Hx as [Hxy Hx].
      apply orb_false_iff in Hf as [Hfy Hf].
      exists x. split; [ exact Hposition | split; [ exact Hx | ] ].
      split; [ exact Hf | ].
      apply Removes_under_let; assumption.
  - destruct (first_read wanted params bound body) as [r0 | ] eqn:Hbody.
    + intros [= <-]. simpl.
      destruct (first_read_removes body wanted params bound r0 Hbody)
        as (x & Hposition & Hx & Hf & Hremoves).
      exists x. split; [ exact Hposition | split; [ exact Hx | ] ].
      split; [ exact Hf | ].
      apply Removes_in_body, Hremoves.
    + destruct (first_read wanted params (handler_params ++ bound) handler)
        as [r0 | ] eqn:Hhandler; [ | discriminate ].
      intros [= <-]. simpl.
      destruct
        (first_read_removes handler wanted params (handler_params ++ bound)
           r0 Hhandler)
        as (x & Hposition & Hx & Hf & Hremoves).
      rewrite among_app in Hx, Hf.
      apply orb_false_iff in Hx as [Hx_params Hx].
      apply orb_false_iff in Hf as [Hf_params Hf].
      exists x. split; [ exact Hposition | split; [ exact Hx | ] ].
      split; [ exact Hf | ].
      apply Removes_in_handler; assumption.
Qed.

(* ======================================================================== *)
(* The other reads of the same parameter by the same primitive              *)
(* ======================================================================== *)

Lemma reads_same_spec (p : unary_primitive) (x : variable) (d : named) :
  reads_same p x d = true -> d = Prim (Unary p) [Var x].
Proof.
  destruct d as [s | pr args | | | | | ]; try discriminate.
  destruct pr as [p' | b | v | hp | rp]; try discriminate.
  destruct args as [ | [x' | sym | c] [ | arg args]]; try discriminate.
  cbn [reads_same]. intros Hsame.
  apply andb_true_iff in Hsame as [Hp Hx].
  apply Pos.eqb_eq in Hx. subst x'.
  unfold prim_eqb in Hp.
  destruct (prim_eq_dec (Unary p') (Unary p)) as [Heq | ];
    [ | discriminate ].
  injection Heq as ->. reflexivity.
Qed.

(* A unary primitive applied to a variable that holds what [f] holds
   has the value that it has applied to [f]. *)
Lemma read_of_same_eval (f : variable) (same : list variable) (u : value)
    (vars : env) (statics : static_env) (d : named) :
  lookup f vars = Some u ->
  (∀ z : variable, among z same = true -> lookup z vars = Some u) ->
  eval_named statics vars (read_of_same f same d) = eval_named statics vars d.
Proof.
  intros Hf Hsame.
  destruct d as [s | pr args | | | | | ]; try reflexivity.
  destruct pr as [p' | b | v | hp | rp]; try reflexivity.
  destruct args as [ | [z | sym | c] [ | arg args]]; try reflexivity.
  cbn [read_of_same]. destruct (among z same) eqn:Hz; [ | reflexivity ].
  cbn [eval_named eval_simples eval_simple]. rewrite Hf, (Hsame z Hz).
  reflexivity.
Qed.

(* Under variables that give [x] a value, and [f] and each of the
   variables [same] what the primitive has on it, the expression with
   the reads shared refines the expression. *)
Fixpoint share_reads_refines (p : unary_primitive) (x f : variable)
    (v u : value) (e : expr) :
  ∀ (same : list variable) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    lookup x vars = Some v ->
    lookup f vars = Some u ->
    gives p v u statics ->
    (∀ z : variable, among z same = true -> lookup z vars = Some u) ->
    refines (State e vars conts statics hp)
      (State (share_reads p x f same e) vars conts statics hp).
Proof.
  destruct e
    as [ y kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros same vars conts statics hp Hx Hf Hgives Hsame;
    cbn [share_reads]; try apply refines_refl.
  - pose proof (read_of_same_eval f same u vars statics d Hf Hsame) as Hd.
    destruct (reads_same p x (read_of_same f same d)) eqn:Hshared.
    + apply reads_same_spec in Hshared. rewrite Hshared in Hd.
      simpl in Hd. rewrite Hx in Hd. simpl in Hd.
      destruct (eval_prim statics (Unary p) [v]) as [w | ] eqn:Hprim;
        [ | apply step_undefined_refines; rewrite step_let, <- Hd;
            reflexivity ].
      pose proof (Hgives statics (statics_extend_refl statics) w Hprim)
        as ->.
      apply let_refines_step;
        [ rewrite <- Hd; cbn [eval_named eval_simple]; exact Hf | ].
      intros w Hw _. rewrite <- Hd in Hw. injection Hw as <-.
      destruct (Pos.eqb_spec y x) as [-> | Hyx]; cbn [orb];
        [ apply refines_refl | ].
      destruct (Pos.eqb_spec y f) as [-> | Hyf]; [ apply refines_refl | ].
      apply (share_reads_refines p x f v u);
        [ rewrite lookup_other by congruence; exact Hx
        | rewrite lookup_other by congruence; exact Hf | exact Hgives | ].
      intros z Hz. destruct (Pos.eq_dec z y) as [-> | Hne];
        [ apply lookup_head | ].
      rewrite lookup_other by exact Hne. apply Hsame.
      rewrite among_cons in Hz. apply orb_true_iff in Hz as [Hz | Hz];
        [ apply Pos.eqb_eq in Hz; contradiction | exact Hz ].
    + apply let_refines_step; [ exact Hd | ].
      intros w _ _.
      destruct (Pos.eqb_spec y x) as [-> | Hyx]; cbn [orb];
        [ apply refines_refl | ].
      destruct (Pos.eqb_spec y f) as [-> | Hyf]; [ apply refines_refl | ].
      apply (share_reads_refines p x f v u);
        [ rewrite lookup_other by congruence; exact Hx
        | rewrite lookup_other by congruence; exact Hf | exact Hgives | ].
      intros z Hz. rewrite among_without in Hz.
      apply andb_true_iff in Hz as [Hz Hy].
      rewrite lookup_other; [ exact (Hsame z Hz) | ].
      intros ->. rewrite among_cons, Pos.eqb_refl in Hy. discriminate.
  - apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (if among x params || among f params
                then handler
                else share_reads p x f (without params same) handler))
            body)
         vars conts statics hp)).
    + destruct (among x params) eqn:Hxp; cbn [orb]; [ apply refines_refl | ].
      destruct (among f params) eqn:Hfp; [ apply refines_refl | ].
      apply (let_cont_handler_refines_where (gives p v u)
        (gives_extend p v u)); [ exact Hgives | ].
      intros vs handler_vars conts0 statics0 hp' Hgives0 _ Hbind.
      apply (share_reads_refines p x f v u); [ | | exact Hgives0 | ].
      * rewrite (bind_params_other params x vs vars handler_vars Hxp Hbind).
        exact Hx.
      * rewrite (bind_params_other params f vs vars handler_vars Hfp Hbind).
        exact Hf.
      * intros z Hz. rewrite among_without in Hz.
        apply andb_true_iff in Hz as [Hz Hparams].
        apply negb_true_iff in Hparams.
        rewrite (bind_params_other params z vs vars handler_vars Hparams
          Hbind).
        exact (Hsame z Hz).
    + apply let_cont_non_recursive_refines_step.
      apply (share_reads_refines p x f v u); assumption.
  - destruct scrutinee as [z | sym | c]; try apply refines_refl.
    destruct (among z same) eqn:Hz; [ | apply refines_refl ].
    apply equivalent_refines, step_eq_equivalent. rewrite !step_switch.
    cbn [eval_simple]. rewrite Hf, (Hsame z Hz). reflexivity.
Qed.

(* The read with the other reads shared is the read, but for what is
   left of the handler. *)
Lemma with_reads_shared_variable (share : bool) (params : list variable)
    (r : read) :
  read_variable (with_reads_shared share params r) = read_variable r.
Proof.
  unfold with_reads_shared. destruct share; [ | reflexivity ].
  destruct (nth_error params (read_position r)); reflexivity.
Qed.

Lemma with_reads_shared_kind (share : bool) (params : list variable)
    (r : read) :
  read_kind (with_reads_shared share params r) = read_kind r.
Proof.
  unfold with_reads_shared. destruct share; [ | reflexivity ].
  destruct (nth_error params (read_position r)); reflexivity.
Qed.

Lemma with_reads_shared_primitive (share : bool) (params : list variable)
    (r : read) :
  read_primitive (with_reads_shared share params r) = read_primitive r.
Proof.
  unfold with_reads_shared. destruct share; [ | reflexivity ].
  destruct (nth_error params (read_position r)); reflexivity.
Qed.

Lemma with_reads_shared_position (share : bool) (params : list variable)
    (r : read) :
  read_position (with_reads_shared share params r) = read_position r.
Proof.
  unfold with_reads_shared. destruct share; [ | reflexivity ].
  destruct (nth_error params (read_position r)); reflexivity.
Qed.

(* What is left of the handler with the other reads shared refines
   what is left of it, where the variable of the read holds what the
   primitive has on the parameter read. *)
Lemma with_reads_shared_refines (share : bool) (params : list variable)
    (r : read) (x : variable) (v u : value) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) :
  nth_error params (read_position r) = Some x ->
  lookup x vars = Some v ->
  lookup (read_variable r) vars = Some u ->
  gives (read_primitive r) v u statics ->
  refines (State (read_without r) vars conts statics hp)
    (State (read_without (with_reads_shared share params r)) vars conts
       statics hp).
Proof.
  intros Hnth Hx Hf Hgives. unfold with_reads_shared.
  destruct share; [ | apply refines_refl ]. rewrite Hnth.
  cbn [read_without].
  apply (share_reads_refines _ _ _ v u); try assumption.
  intros z Hz. discriminate.
Qed.

(* A handler refines what is left of it with the other reads shared,
   where the new parameter holds what the read gives. *)
Lemma shared_read_fits_refines (share : bool) (params : list variable)
    (r : read) (x : variable) (handler : expr) (def_vars bound : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (vs : list value)
    (u : value) :
  removes (read_variable r) (read_kind r) (read_primitive r) x handler
    (read_without r) ->
  var_free_b (read_variable r) handler = false ->
  Pos.eqb x (read_variable r) = false ->
  distinct params = true ->
  nth_error params (read_position r) = Some x ->
  bind_params params vs def_vars = Some bound ->
  read_fits (read_position r) (read_primitive r) statics vs u ->
  refines (State handler bound conts statics hp)
    (State (read_without (with_reads_shared share params r))
       ((read_variable r, u) :: bound) conts statics hp).
Proof.
  intros Hremoves Hfree Hne Hdistinct Hx Hbind Hfits.
  apply (refines_trans _
    (State (read_without r) ((read_variable r, u) :: bound) conts statics
       hp)).
  - exact (read_fits_refines (read_variable r) (read_kind r)
      (read_primitive r) x (read_position r) params handler (read_without r)
      def_vars bound conts statics hp vs u Hremoves Hfree Hne Hdistinct Hx
      Hbind Hfits).
  - destruct Hfits as (v & Hv & Hgives).
    apply (with_reads_shared_refines share params r x v u);
      [ exact Hx | | apply lookup_head | exact Hgives ].
    rewrite lookup_other by (apply Pos.eqb_neq; exact Hne).
    exact (bind_params_nth params vs def_vars bound (read_position r) x v
      Hdistinct Hbind Hx Hv).
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* One read made a parameter. *)
Lemma unbox_read_refines
    (extend : facts -> list simple -> option (list simple))
    (fits : static_env -> list value -> value -> Prop) (known : facts)
    (k : continuation) (params params' : list variable)
    (metadata metadata' : handler_metadata) (handler handler' body body' : expr)
    (r : read) (x : variable) :
  unbox_read extend known k params metadata handler body r
  = Some (params', metadata', handler', body') ->
  position x params = Some (read_position r) ->
  (∀ (known : facts) (vars : env) (statics : static_env) (hp : heap)
     (args args' : list simple) (vs : list value) (v : value),
     facts_hold known vars statics ->
     extend known args = Some args' ->
     eval_simples vars args = Some vs ->
     nth_error vs (read_position r) = Some v ->
     (∃ u : value,
        eval_simples vars args' = Some (vs ++ [u])
        ∧ kind_of_value u = read_kind r
        ∧ fits statics vs u)
     ∨ (∀ (bound : env) (conts : cont_env),
          lookup x bound = Some v ->
          ∃ n : nat, run n (State handler bound conts statics hp) = Undefined))
  ->
  (var_free_b (read_variable r) handler = false ->
   Pos.eqb x (read_variable r) = false ->
   distinct params = true ->
   among (read_variable r) params = false ->
   ∀ (def_vars bound : env) (conts : cont_env) (statics : static_env)
     (hp : heap)
     (vs : list value) (u : value),
     bind_params params vs def_vars = Some bound ->
     fits statics vs u ->
     refines (State handler bound conts statics hp)
       (State (read_without r) ((read_variable r, u) :: bound) conts
          statics hp))
  ->
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap),
    facts_hold known vars statics ->
    refines
      (State (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp)
      (State
         (Let_cont_non_recursive k (Handler params' metadata' handler') body')
         vars conts statics hp).
Proof.
  intros Hread Hposition Hextend Halike vars conts statics hp Hfacts.
  unfold unbox_read in Hread.
  destruct (may_take params metadata (read_variable r)) eqn:Htake;
    [ | discriminate ].
  destruct (var_free_b (read_variable r) handler) eqn:Hfree;
    [ discriminate | ].
  simpl in Hread.
  destruct (extend_applications_by extend known k body) as [extended | ]
    eqn:Hbody; [ | discriminate ].
  injection Hread as <- <- <- <-.
  unfold may_take in Htake.
  apply andb_true_iff in Htake as [Htake Hfresh].
  apply andb_true_iff in Htake as [Htake _].
  apply andb_true_iff in Htake as [_ Hdistinct].
  apply negb_true_iff in Hfresh.
  assert (Hne : Pos.eqb x (read_variable r) = false).
  { destruct (Pos.eqb_spec x (read_variable r)) as [Heq | _];
      [ | reflexivity ].
    rewrite <- Heq, (position_among x params _ Hposition) in Hfresh.
    discriminate. }
  exact (unbox_parameter_refines k (read_position r) params metadata
    (with_one_more metadata (read_kind r)) (read_variable r) x (read_kind r)
    handler (read_without r) extend fits Hdistinct
    (position_nth x params _ Hposition) eq_refl Hextend
    (Halike eq_refl Hne Hdistinct Hfresh) known body extended vars conts
    statics hp Hbody Hfacts).
Qed.

(* The read that is taken next is a first read of some sort. *)
Lemma next_read_first (params : list variable) (handler : expr) (r : read) :
  next_read params handler = Some r ->
  ∃ wanted : unary_primitive -> bool,
    first_read wanted params [] handler = Some r.
Proof.
  unfold next_read.
  destruct (first_read reads_tag params [] handler) as [r0 | ] eqn:Htag;
    [ intros [= <-]; exists reads_tag; exact Htag | ].
  destruct (first_read reads_constant_constructor params [] handler)
    as [r0 | ] eqn:Hconstant;
    [ intros [= <-]; exists reads_constant_constructor; exact Hconstant | ].
  destruct (first_read reads_is_int params [] handler) as [r0 | ] eqn:His_int;
    [ intros [= <-]; exists reads_is_int; exact His_int | ].
  intros Hcontents. exists reads_contents. exact Hcontents.
Qed.

(* The [Let] that the handler begins with, made a parameter. *)
Lemma head_step_refines (known : facts) (k : continuation)
    (params params' : list variable) (metadata metadata' : handler_metadata)
    (handler handler' body body' : expr) (head : read) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (share : bool) :
  head_read params handler = Some head ->
  unbox_read
    (fun known => extend_args known (read_position head) (read_primitive head))
    known k params metadata handler body (with_reads_shared share params head)
  = Some (params', metadata', handler', body') ->
  facts_hold known vars statics ->
  refines
    (State (Let_cont_non_recursive k (Handler params metadata handler) body)
       vars conts statics hp)
    (State
       (Let_cont_non_recursive k (Handler params' metadata' handler') body')
       vars conts statics hp).
Proof.
  intros Hhead Hstep Hfacts. unfold head_read in Hhead.
  destruct handler as [ f kind d rest | | | | | | | ]; try discriminate.
  destruct (param_read reads_any params [] f d) as [[p i] | ] eqn:Hread;
    [ | discriminate ].
  simpl in Hhead. injection Hhead as <-.
  destruct (param_read_spec reads_any params [] f d p i Hread)
    as (x & -> & Hposition & _ & _).
  pose proof
    (unbox_read_refines _ (read_fits i p) known k params params' metadata
       metadata' _ handler' body body'
       (with_reads_shared share params (Read f kind p i rest)) x Hstep)
    as Hgoal.
  rewrite with_reads_shared_position, with_reads_shared_kind,
    with_reads_shared_variable in Hgoal.
  exact (Hgoal Hposition (extend_args_gives i p f x kind rest)
    (fun Hfree Hne Hdistinct _ def_vars bound conts0 statics0 hp vs u Hbind
       Hfits =>
       shared_read_fits_refines share params (Read f kind p i rest) x _ def_vars
         bound conts0 statics0 _ vs u (Removes_here f kind p x rest) Hfree
         Hne Hdistinct (position_nth x params i Hposition) Hbind Hfits)
    vars conts statics hp Hfacts).
Qed.

(* A read further down, made a parameter. *)
Lemma certain_step_refines (known : facts) (k : continuation)
    (params params' : list variable) (metadata metadata' : handler_metadata)
    (handler handler' body body' : expr) (r : read) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (variant : bool)
    (tests : list variable) (share : bool) :
  next_read params handler = Some r ->
  unbox_read
    (fun known =>
       extend_args_certain variant tests params handler
         (read_without (with_reads_shared share params r)) known
         (read_position (with_reads_shared share params r))
         (read_primitive (with_reads_shared share params r))
         (read_kind (with_reads_shared share params r)))
    known k params metadata handler body (with_reads_shared share params r)
  = Some (params', metadata', handler', body') ->
  facts_hold known vars statics ->
  refines
    (State (Let_cont_non_recursive k (Handler params metadata handler) body)
       vars conts statics hp)
    (State
       (Let_cont_non_recursive k (Handler params' metadata' handler') body')
       vars conts statics hp).
Proof.
  intros Hnext Hstep Hfacts.
  destruct (next_read_first params handler r Hnext) as [wanted Hfirst].
  destruct (first_read_removes handler wanted params [] r Hfirst)
    as (x & Hposition & _ & _ & Hremoves).
  rewrite with_reads_shared_position, with_reads_shared_kind,
    with_reads_shared_primitive in Hstep.
  pose proof
    (unbox_read_refines _
       (certain_fits params handler
          (read_without (with_reads_shared share params r)) (read_position r)
          (read_primitive r))
       known k params params' metadata metadata' handler handler' body body'
       (with_reads_shared share params r) x Hstep)
    as Hgoal.
  rewrite with_reads_shared_position, with_reads_shared_kind,
    with_reads_shared_variable in Hgoal.
  apply Hgoal; [ exact Hposition | | | exact Hfacts ].
  - intros known0 vars0 statics0 hp' args args' vs v Hfacts0 Hextend Hvs Hv.
    left.
    exact (extend_args_certain_gives variant tests params handler
      (read_without (with_reads_shared share params r)) (read_position r)
      (read_primitive r) (read_kind r) known0 vars0 statics0 args args' vs v
      Hfacts0 Hextend Hvs Hv).
  - intros Hfree Hne Hdistinct Hfresh def_vars bound conts0 statics0 hp' vs
      u Hbind [Hfits | (consts & Hnever & Hvalues)].
    + exact (shared_read_fits_refines share params r x handler def_vars bound
        conts0 statics0 hp' vs u Hremoves Hfree Hne Hdistinct
        (position_nth x params _ Hposition) Hbind Hfits).
    + apply (refines_trans _
        (State handler ((read_variable r, u) :: bound) conts0 statics0 hp'));
        [ apply unread_refines, Hfree | ].
      apply (never_reads_refines consts handler
        (read_without (with_reads_shared share params r)) _ conts0 statics0 _
        Hnever).
      intros t c Ht. destruct (Hvalues t c Ht) as (j & Hparam & Hvalue).
      rewrite lookup_other.
      * exact (bind_params_nth params vs def_vars bound j t _ Hdistinct Hbind
          Hparam Hvalue).
      * intros ->. apply nth_error_In, among_In in Hparam. congruence.
Qed.

#[local] Arguments loads_of_two_tags : simpl never.
#[local] Arguments with_reads_shared : simpl never.
#[local] Arguments shares_reads : simpl never.

(* The reads of a handler, one after the other. Where the pass stops
   at a load to which blocks of more than one tag may come, the term
   is as it was. *)
Lemma unbox_reads_refines (fuel : nat) :
  ∀ (known : facts) (k : continuation) (made : list (variable * nat))
    (above : list (nat * list nat)) (params : list variable)
    (metadata : handler_metadata) (handler body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap),
    facts_hold known vars statics ->
    refines
      (State (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp)
      (State (unbox_reads fuel known k made above params metadata handler body)
         vars conts statics hp).
Proof.
  induction fuel as [ | fuel IH];
    intros known k made above params metadata handler body vars conts statics
      hp Hfacts;
    simpl; [ apply refines_refl | ].
  destruct (next_read params handler) as [r0 | ] eqn:Hnext; simpl;
    [ | apply refines_refl ].
  set (share :=
    shares_reads known k (tests_of (read_position r0) made above) params
      handler body r0).
  set (r := with_reads_shared share params r0).
  destruct
    (loads_of_two_tags known k (tests_of (read_position r) made above) params
       handler body r);
    [ apply refines_refl | ].
  set (made' :=
    if reads_test (read_primitive r)
    then (read_variable r, read_position r) :: made
    else made).
  set (above' :=
    (length params, length params :: boxes_of (read_position r) above)
    :: above).
  set (variant := tested_is_int params (read_position r) handler).
  set (tests := tests_of (read_position r) made above).
  assert (Hcertain :
    refines
      (State (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp)
      (State
         match
           unbox_read
             (fun known =>
                extend_args_certain variant tests params handler
                  (read_without r) known (read_position r) (read_primitive r)
                  (read_kind r))
             known k params metadata handler body r
         with
         | Some (params0, metadata0, handler0, body0) =>
           unbox_reads fuel known k made' above' params0 metadata0 handler0
             body0
         | None =>
           Let_cont_non_recursive k (Handler params metadata handler) body
         end
         vars conts statics hp)).
  { destruct
      (unbox_read
         (fun known =>
            extend_args_certain variant tests params handler (read_without r)
              known (read_position r) (read_primitive r) (read_kind r))
         known k params metadata handler body r)
      as [[[[params' metadata'] handler'] body'] | ] eqn:Hstep;
      [ | apply refines_refl ].
    eapply refines_trans; [ | apply IH; exact Hfacts ].
    exact (certain_step_refines known k params params' metadata metadata'
      handler handler' body body' r0 vars conts statics _ variant tests share
      Hnext Hstep Hfacts). }
  destruct (head_read params handler) as [head | ] eqn:Hhead;
    [ | exact Hcertain ].
  destruct (Pos.eqb (read_variable head) (read_variable r));
    [ | exact Hcertain ].
  destruct
    (unbox_read
       (fun known =>
          extend_args known (read_position head) (read_primitive head))
       known k params metadata handler body
       (with_reads_shared share params head))
    as [[[[params' metadata'] handler'] body'] | ] eqn:Hstep;
    [ | apply refines_refl ].
  eapply refines_trans; [ | apply IH; exact Hfacts ].
  exact (head_step_refines known k params params' metadata metadata' handler
    handler' body body' head vars conts statics _ share Hhead Hstep Hfacts).
Qed.

(* The handler of each non-recursive [Let_cont] and its body, by the
   induction, with what is known there; then the reads of the
   handler. *)
Fixpoint unbox_variants_knowing_refines (e : expr) :
  ∀ (known : facts) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap),
    facts_hold known vars statics ->
    refines (State e vars conts statics hp)
      (State (unbox_variants_knowing known e) vars conts statics hp).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap
       | ac | scrutinee arms | ];
    intros known vars conts statics hp Hfacts; simpl.
  - (* Let *)
    apply let_refines_step; [ reflexivity | ].
    intros v Hd _. apply unbox_variants_knowing_refines.
    exact (facts_hold_under_let known x d vars statics v Hfacts Hd).
  - (* Let_static *)
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings; [ | exact I ].
    apply unbox_variants_knowing_refines.
    exact (facts_hold_under_statics known vars statics bindings extended
      Hfacts Hbindings).
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler].
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (unbox_variants_knowing (facts_under_params known params)
                  handler))
            body)
         vars conts statics hp)).
    + apply (let_cont_handler_refines_where (facts_hold known vars)
        (facts_hold_extend known vars));
        [ exact Hfacts | ].
      intros vs handler_vars conts0 statics0 hp' Hfacts0 _ Hbind.
      apply unbox_variants_knowing_refines.
      exact (facts_hold_bind_params statics0 params known vs vars
        handler_vars Hfacts0 Hbind).
    + apply (refines_trans _
        (State
           (Let_cont_non_recursive k
              (Handler params metadata
                 (unbox_variants_knowing (facts_under_params known params)
                    handler))
              (unbox_variants_knowing known body))
           vars conts statics hp)).
      * apply let_cont_non_recursive_refines_step.
        apply unbox_variants_knowing_refines, Hfacts.
      * apply unbox_reads_refines, Hfacts.
  - (* Let_cont_recursive *)
    apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
    apply unbox_variants_knowing_refines, Hfacts.
  - apply refines_refl.
  - apply refines_refl.
  - apply refines_refl.
  - apply refines_refl.
Qed.

Theorem unbox_variant_params_refines : unbox_variant_params_refines_statement.
Proof.
  intros e vars conts statics hp. apply unbox_variants_knowing_refines.
  apply no_facts_hold.
Qed.
