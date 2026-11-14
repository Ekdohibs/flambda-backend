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


(* Proofs of the statements of FlambdaSimplifyUsesSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextWhereSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinProofs FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyPrimitivesProofs FlambdaSimplifyProjectionsSpec
  FlambdaSimplifyCodeEnvSpec
  FlambdaSimplifyCodeEnvProofs FlambdaSimplifyHandlersSpec
  FlambdaSimplifyHandlersProofs FlambdaSimplifyPassSpec
  FlambdaSimplifyUsesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The tests                                                                *)
(* ======================================================================== *)

Lemma simple_in_terms_b_sound (tenv : Typing_env.t) (s : simple) :
  simple_in_terms_b tenv s = true -> simple_in_terms tenv s.
Proof. intros Hb x ->. exact Hb. Qed.

Lemma simple_of_kind_b_sound (tenv : Typing_env.t) (s : simple)
    (kind : flambda_kind) :
  simple_of_kind_b tenv s kind = true -> simple_of_kind tenv s kind.
Proof.
  destruct s as [x | sym | c]; simpl.
  - destruct (Typing_env.find tenv (Name_var x) (Some kind)) as [ty | ];
      [ | discriminate ].
    intros _. exists ty. reflexivity.
  - destruct (Typing_env.find tenv (Name_symbol sym) (Some kind))
      as [ty | ]; [ | discriminate ].
    intros _. exists ty. reflexivity.
  - intros Hkind. exact (proj1 (kind_eqb_eq _ _) Hkind).
Qed.

(* ======================================================================== *)
(* A valuation that shows that a state satisfies an environment             *)
(* ======================================================================== *)

(* The statements are of states ([env_satisfies_strictly]); the
   contract of the join asks for one valuation that satisfies two
   environments, so the lemmas keep the valuation in hand. *)
Definition shows (tenv : Typing_env.t) (statics : static_env)
    (valuation vars : env) : Prop :=
  bound_in_terms tenv vars
  ∧ typing_env_satisfied_strictly statics valuation tenv
  ∧ agrees_in_terms tenv valuation vars.

Lemma shows_satisfies (tenv : Typing_env.t) (statics : static_env)
    (valuation vars : env) :
  shows tenv statics valuation vars ->
  env_satisfies_strictly tenv statics vars.
Proof.
  intros [Hbound [Hsat Hagrees]]. split; [ exact Hbound | ].
  exists valuation. split; assumption.
Qed.

Lemma satisfies_shows (tenv : Typing_env.t) (statics : static_env)
    (vars : env) :
  env_satisfies_strictly tenv statics vars ->
  ∃ valuation : env, shows tenv statics valuation vars.
Proof.
  intros [Hbound [valuation [Hsat Hagrees]]]. exists valuation.
  split; [ exact Hbound | split; assumption ].
Qed.

(* A variable defined, and bound to a value of its kind. *)
Lemma define_shows (tenv : Typing_env.t) (x : variable) (kind : flambda_kind)
    (v : value) (statics : static_env) (valuation vars : env) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  kind_of_value v = kind ->
  shows tenv statics valuation vars ->
  shows (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
    statics ((x, v) :: valuation) ((x, v) :: vars).
Proof.
  intros Hwf Hfresh Hkind [Hbound [Hsatisfied Hagrees]].
  split; [ | split ].
  - intros y Hy.
    apply (add_variable_definition_in_scope tenv x kind y Hwf) in Hy.
    destruct Hy as [-> | [Hne Hy]].
    + exists v. apply lookup_head.
    + destruct (Hbound y Hy) as [w Hw]. exists w.
      exact (eq_trans (lookup_other y x v vars Hne) Hw).
  - apply add_variable_definition_satisfied_bind;
      [ exact Hsatisfied | | exact Hkind ].
    exact (well_formed_not_mentioned tenv x Hwf Hfresh).
  - intros y Hy.
    apply (add_variable_definition_in_scope tenv x kind y Hwf) in Hy.
    destruct Hy as [-> | [Hne Hy]].
    + exact (eq_trans (lookup_head x v valuation)
               (eq_sym (lookup_head x v vars))).
    + exact (eq_trans (lookup_other y x v valuation Hne)
               (eq_trans (Hagrees y Hy)
                  (eq_sym (lookup_other y x v vars Hne)))).
Qed.

(* Two [Simple]s that have the same value in the machine made equal. *)
Lemma alias_shows (tenv : Typing_env.t) (c1 c2 : simple)
    (result : Typing_env.add_alias_result) (statics : static_env)
    (valuation vars : env) :
  (∀ x : variable, c1 = Var x ∨ c2 = Var x -> in_scope_in_terms tenv x) ->
  Typing_env.add_alias tenv c1 c2 = Some (Ok result) ->
  eval_simple vars c1 = eval_simple vars c2 ->
  shows tenv statics valuation vars ->
  shows (Typing_env.add_alias_result_t result) statics valuation vars.
Proof.
  intros Hscope Hadd Heval [Hbound [Hsatisfied Hagrees]].
  split; [ | split ].
  - intros y Hy. apply Hbound.
    apply (add_alias_in_scope tenv c1 c2 result y Hadd). exact Hy.
  - apply (add_alias_satisfied (fun _ => False) statics valuation tenv c1
             c2 result Hsatisfied Hadd).
    rewrite (agrees_eval_simple tenv valuation vars c1 Hagrees),
      (agrees_eval_simple tenv valuation vars c2 Hagrees);
      [ exact Heval | | ];
      intros x ->; apply Hscope; [ right | left ]; reflexivity.
  - intros y Hy. apply Hagrees.
    apply (add_alias_in_scope tenv c1 c2 result y Hadd). exact Hy.
Qed.

(* ======================================================================== *)
(* The environment of an arm                                                *)
(* ======================================================================== *)

Lemma arm_env_in_scope (tenv : Typing_env.t) (scrutinee : simple) (d : Z)
    (y : variable) :
  in_scope_in_terms (arm_env tenv scrutinee d) y <-> in_scope_in_terms tenv y.
Proof.
  unfold arm_env.
  destruct (simplify_simple tenv scrutinee) as [[ty [c | sym | const]] | ];
    try reflexivity.
  destruct (simple_in_terms_b tenv (Var c)
            && simple_of_kind_b tenv (Var c) naked_immediate_kind);
    [ | reflexivity ].
  destruct (Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)))
    as [[ | result | ] | ] eqn:Hadd; try reflexivity.
  exact (add_alias_in_scope tenv _ _ result y Hadd).
Qed.

Lemma arm_env_shows (tenv : Typing_env.t) (scrutinee : simple) (d : Z)
    (statics : static_env) (valuation vars : env) :
  shows tenv statics valuation vars ->
  eval_simple vars scrutinee = Some (Value_const (Naked_immediate d)) ->
  shows (arm_env tenv scrutinee d) statics valuation vars.
Proof.
  intros Hshows Heval. unfold arm_env.
  destruct (simplify_simple tenv scrutinee) as [[ty canonical] | ] eqn:Hs;
    [ | exact Hshows ].
  destruct canonical as [c | sym | const]; try exact Hshows.
  destruct (simple_in_terms_b tenv (Var c)
            && simple_of_kind_b tenv (Var c) naked_immediate_kind)
    eqn:Htests; [ | exact Hshows ].
  apply andb_true_iff in Htests. destruct Htests as [Hterms _].
  destruct (Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)))
    as [[ | result | ] | ] eqn:Hadd; try exact Hshows.
  apply (alias_shows tenv (Var c) (Const (Naked_immediate d)) result statics
           valuation vars);
    [ | exact Hadd | | exact Hshows ].
  - intros x [[= <-] | Hx]; [ | discriminate ].
    exact (simple_in_terms_b_sound tenv (Var c) Hterms c eq_refl).
  - destruct (simplify_simple_answer tenv scrutinee (Var c) ty Hs)
      as [_ Hcanonical].
    rewrite (canonical_in_terms_sound tenv statics vars scrutinee (Var c)
               (shows_satisfies tenv statics valuation vars Hshows)
               Hcanonical).
    exact Heval.
Qed.

Theorem arm_env_satisfied : arm_env_satisfied_statement.
Proof.
  intros tenv scrutinee d statics vars Hsat Heval.
  destruct (satisfies_shows tenv statics vars Hsat) as [valuation Hshows].
  exact (shows_satisfies _ statics valuation vars
           (arm_env_shows tenv scrutinee d statics valuation vars Hshows
              Heval)).
Qed.

(* ======================================================================== *)
(* The environment made for an application                                  *)
(* ======================================================================== *)

(* The parameters of a list of bindings. *)
Definition bound_params (bindings : list (variable * flambda_kind * simple))
    : list variable :=
  map (fun binding => fst (fst binding)) bindings.

Lemma not_among_cons (x : variable) (xs : list variable) (s : simple) :
  not_among (x :: xs) s = true -> s <> Var x ∧ not_among xs s = true.
Proof.
  destruct s as [y | sym | c];
    [ | intros _; split; [ discriminate | reflexivity ].. ].
  unfold not_among. rewrite among_cons. intros Hnot.
  apply negb_true_iff in Hnot.
  apply orb_false_iff in Hnot. destruct Hnot as [Hne Hrest].
  split; [ | rewrite Hrest; reflexivity ].
  intros [= ->]. rewrite Pos.eqb_refl in Hne. discriminate.
Qed.

Lemma eval_simple_other (x : variable) (v : value) (vars : env) (s : simple) :
  s <> Var x -> eval_simple ((x, v) :: vars) s = eval_simple vars s.
Proof.
  destruct s as [y | sym | c]; [ | reflexivity.. ]. intros Hne. simpl.
  apply lookup_other. congruence.
Qed.

(* The parameters defined and made aliases, and bound as the machine
   binds them: the valuation that shows it binds them in the same
   way. *)
Lemma bind_params_to_args_shows (aliased : simple -> bool)
    (bindings : list (variable * flambda_kind * simple)) :
  ∀ (tenv target : Typing_env.t) (statics : static_env)
    (valuation vars vars' : env) (vs : list value),
    bind_params_to_args aliased tenv bindings = Some target ->
    (∀ c : simple,
      aliased c = true -> not_among (bound_params bindings) c = true) ->
    Forall2
      (fun binding v =>
         kind_of_value v = snd (fst binding)
         ∧ (aliased (snd binding) = true ->
            eval_simple vars (snd binding) = Some v))
      bindings vs ->
    bind_params (bound_params bindings) vs vars = Some vars' ->
    shows tenv statics valuation vars ->
    ∃ valuation' : env,
      bind_params (bound_params bindings) vs valuation = Some valuation'
      ∧ shows target statics valuation' vars'.
Proof.
  induction bindings as [ | [[x kind] canonical] bindings IH];
    intros tenv target statics valuation vars vars' vs Hdefined Haliased
      Hvalues Hbind Hshows.
  - inversion Hvalues; subst. simpl in Hdefined, Hbind.
    injection Hdefined as <-. injection Hbind as <-.
    exists valuation. split; [ reflexivity | exact Hshows ].
  - inversion Hvalues as [ | binding v bindings' vs' [Hkind Hvalue] Hrest];
      subst.
    cbn [bound_params map fst snd bind_params] in Hbind |- *.
    cbn [bind_params_to_args] in Hdefined.
    cbn [fst snd] in Hkind, Hvalue.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_prop in Htests. destruct Htests as [Hwf Hfresh].
    apply negb_true_iff in Hfresh.
    pose proof (define_shows tenv x kind v statics valuation vars Hwf Hfresh
                  Hkind Hshows) as Hdefined_shows.
    apply (IH _ target statics ((x, v) :: valuation) ((x, v) :: vars) vars'
             vs' Hdefined).
    + intros c Hc. exact (proj2 (not_among_cons x _ c (Haliased c Hc))).
    + eapply Forall2_impl; [ | exact Hrest ].
      intros binding' v' [Hkind' Hvalue']. split; [ exact Hkind' | ].
      intros Hc. rewrite eval_simple_other; [ exact (Hvalue' Hc) | ].
      exact (proj1 (not_among_cons x _ _ (Haliased _ Hc))).
    + exact Hbind.
    + destruct (aliased canonical && simple_in_terms_b tenv canonical
                && simple_of_kind_b tenv canonical kind) eqn:Halias;
        [ | exact Hdefined_shows ].
      apply andb_true_iff in Halias. destruct Halias as [Halias _].
      apply andb_true_iff in Halias. destruct Halias as [Hc Hterms].
      destruct (Typing_env.add_alias
                  (Typing_env.add_variable_definition tenv x kind
                     Name_mode.normal)
                  (Var x) canonical)
        as [[ | result | ] | ] eqn:Hadd; try exact Hdefined_shows.
      assert (Hother : canonical <> Var x).
      { intros ->.
        exact (not_bound_not_in_scope tenv x x Hfresh
                 (simple_in_terms_b_sound tenv (Var x) Hterms x eq_refl)
                 eq_refl). }
      apply (alias_shows
               (Typing_env.add_variable_definition tenv x kind
                  Name_mode.normal)
               (Var x) canonical result statics ((x, v) :: valuation)
               ((x, v) :: vars));
        [ | exact Hadd | | exact Hdefined_shows ].
      * intros y [[= <-] | ->];
          apply (add_variable_definition_in_scope tenv x kind _ Hwf);
          [ left; reflexivity | right ].
        split; [ congruence | ].
        exact (simple_in_terms_b_sound tenv (Var y) Hterms y eq_refl).
      * rewrite (eval_simple_other x v vars canonical Hother), (Hvalue Hc).
        simpl. apply lookup_head.
Qed.

(* The values of a list of [Simple]s, one by one. *)
Lemma eval_simples_each (vars : env) (args : list simple) :
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    Forall2 (fun arg v => eval_simple vars arg = Some v) args vs.
Proof.
  induction args as [ | arg args IH]; intros vs; simpl.
  - intros [= <-]. apply Forall2_nil.
  - destruct (eval_simple vars arg) as [v | ] eqn:Harg; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    intros [= <-]. apply Forall2_cons; [ exact Harg | ]. apply IH.
    reflexivity.
Qed.

(* The bindings made of the parameters, their kinds and the canonical
   elements of the arguments. *)
Lemma bindings_of_params (aliased : simple -> bool) (vars : env)
    (params : list variable) :
  ∀ (kinds : list flambda_kind) (canonicals : list simple) (vs : list value),
    length kinds = length params ->
    length canonicals = length params ->
    map kind_of_value vs = kinds ->
    Forall2 (fun canonical v => eval_simple vars canonical = Some v)
      canonicals vs ->
    bound_params (combine (combine params kinds) canonicals) = params
    ∧ Forall2
        (fun binding v =>
           kind_of_value v = snd (fst binding)
           ∧ (aliased (snd binding) = true ->
              eval_simple vars (snd binding) = Some v))
        (combine (combine params kinds) canonicals) vs.
Proof.
  induction params as [ | x params IH];
    intros [ | kind kinds] [ | canonical canonicals] vs Hkinds Hcanonicals
      Hvs Hvalues;
    simpl in Hkinds, Hcanonicals; try discriminate.
  - inversion Hvalues; subst. split; [ reflexivity | apply Forall2_nil ].
  - inversion Hvalues as [ | c v canonicals' vs' Hvalue Hrest]; subst.
    simpl in Hvs. injection Hvs as Hkind Hvs.
    destruct (IH kinds canonicals vs' ltac:(congruence) ltac:(congruence)
                Hvs Hrest)
      as [Hparams Hbindings].
    split.
    + cbn [combine bound_params map fst]. f_equal. exact Hparams.
    + cbn [combine]. apply Forall2_cons; [ | exact Hbindings ].
      split; [ exact Hkind | intros _; exact Hvalue ].
Qed.

Lemma use_env_shows (scope tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (args : list simple) (target : Typing_env.t)
    (statics : static_env) (valuation vars : env) (vs : list value)
    (handler_vars : env) :
  use_env scope tenv params metadata args = Some target ->
  shows tenv statics valuation vars ->
  eval_simples vars args = Some vs ->
  map kind_of_value vs = hm_params_kinds metadata ->
  bind_params params vs vars = Some handler_vars ->
  ∃ valuation' : env,
    bind_params params vs valuation = Some valuation'
    ∧ shows target statics valuation' handler_vars.
Proof.
  unfold use_env. intros Huse Hshows Hargs Hkinds Hbind.
  destruct (simplify_simples tenv args) as [answers | ] eqn:Hanswers;
    [ | discriminate ].
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params)
            && Nat.eqb (length answers) (length params)) eqn:Hlengths;
    [ | discriminate ].
  apply andb_true_iff in Hlengths. destruct Hlengths as [Hlength Hlength'].
  apply Nat.eqb_eq in Hlength, Hlength'.
  destruct (simplify_simples_answers tenv args answers Hanswers)
    as [_ Hcanonicals].
  pose proof (canonical_in_terms_simples tenv statics vars args
                (map snd answers)
                (shows_satisfies tenv statics valuation vars Hshows)
                Hcanonicals) as Hsame.
  rewrite Hargs in Hsame.
  destruct (bindings_of_params
              (fun canonical =>
                 simple_in_terms_b scope canonical
                 && not_among params canonical)
              vars params (hm_params_kinds metadata) (map snd answers) vs
              Hlength ltac:(rewrite length_map; exact Hlength') Hkinds
              (eval_simples_each vars _ vs Hsame))
    as [Hparams Hbindings].
  rewrite <- Hparams in Hbind |- *.
  apply (bind_params_to_args_shows _ _ tenv target statics valuation vars
           handler_vars vs Huse);
    [ | exact Hbindings | exact Hbind | exact Hshows ].
  intros c Hc. apply andb_true_iff in Hc. rewrite Hparams. exact (proj2 Hc).
Qed.

Theorem use_env_satisfied : use_env_satisfied_statement.
Proof.
  intros scope tenv params metadata args target statics vars vs handler_vars
    Huse Hsat Hargs Hkinds Hbind.
  destruct (satisfies_shows tenv statics vars Hsat) as [valuation Hshows].
  destruct (use_env_shows scope tenv params metadata args target statics
              valuation vars vs handler_vars Huse Hshows Hargs Hkinds Hbind)
    as [valuation' [_ Hshows']].
  exact (shows_satisfies target statics valuation' handler_vars Hshows').
Qed.

(* ======================================================================== *)
(* The environment of the handler where it is defined                       *)
(* ======================================================================== *)

Lemma define_variables_checked_shows
    (definitions : list (variable * flambda_kind)) :
  ∀ (tenv tenv' : Typing_env.t) (statics : static_env)
    (valuation vars vars' : env) (vs : list value),
    define_variables_checked tenv definitions = Some tenv' ->
    Forall2 (fun definition v => kind_of_value v = snd definition)
      definitions vs ->
    bind_params (map fst definitions) vs vars = Some vars' ->
    shows tenv statics valuation vars ->
    ∃ valuation' : env,
      bind_params (map fst definitions) vs valuation = Some valuation'
      ∧ shows tenv' statics valuation' vars'.
Proof.
  induction definitions as [ | [x kind] definitions IH];
    intros tenv tenv' statics valuation vars vars' vs Hdefined Hkinds Hbind
      Hshows.
  - inversion Hkinds; subst. simpl in Hdefined, Hbind.
    injection Hdefined as <-. injection Hbind as <-.
    exists valuation. split; [ reflexivity | exact Hshows ].
  - inversion Hkinds as [ | definition v definitions' vs' Hkind Hrest];
      subst.
    simpl in Hbind. cbn [define_variables_checked] in Hdefined.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_prop in Htests. destruct Htests as [Hwf Hfresh].
    apply negb_true_iff in Hfresh.
    exact (IH _ tenv' statics ((x, v) :: valuation) ((x, v) :: vars) vars'
             vs' Hdefined Hrest Hbind
             (define_shows tenv x kind v statics valuation vars Hwf Hfresh
                Hkind Hshows)).
Qed.

Lemma handler_env_shows (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (tenv_handler : Typing_env.t)
    (statics : static_env) (valuation vars : env) (vs : list value)
    (handler_vars : env) :
  handler_env tenv params metadata = Some tenv_handler ->
  shows tenv statics valuation vars ->
  map kind_of_value vs = hm_params_kinds metadata ->
  bind_params params vs vars = Some handler_vars ->
  ∃ valuation' : env,
    bind_params params vs valuation = Some valuation'
    ∧ shows tenv_handler statics valuation' handler_vars.
Proof.
  intros Henv Hshows Hkinds Hbind. unfold handler_env in Henv.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params))
    eqn:Hlength;
    [ | discriminate ].
  apply Nat.eqb_eq in Hlength.
  destruct (define_variables_checked_shows
              (combine params (hm_params_kinds metadata)) tenv tenv_handler
              statics valuation vars handler_vars vs Henv)
    as [valuation' [Hbind' Hshows']].
  - rewrite <- Hkinds. apply kinds_of_values.
    exact (bind_params_length params vs vars handler_vars Hbind).
  - rewrite (map_fst_combine params _ Hlength). exact Hbind.
  - exact Hshows.
  - exists valuation'. rewrite (map_fst_combine params _ Hlength) in Hbind'.
    split; [ exact Hbind' | exact Hshows' ].
Qed.

(* ======================================================================== *)
(* The variables of terms                                                   *)
(* ======================================================================== *)

Lemma define_variables_checked_in_scope
    (definitions : list (variable * flambda_kind)) :
  ∀ (tenv tenv' : Typing_env.t) (y : variable),
    define_variables_checked tenv definitions = Some tenv' ->
    in_scope_in_terms tenv' y ->
    y ∈ map fst definitions ∨ in_scope_in_terms tenv y.
Proof.
  induction definitions as [ | [x kind] definitions IH];
    intros tenv tenv' y Hdefined Hy.
  - simpl in Hdefined. injection Hdefined as <-. right. exact Hy.
  - cbn [define_variables_checked] in Hdefined.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_prop in Htests. destruct Htests as [Hwf _].
    destruct (IH _ tenv' y Hdefined Hy) as [Hin | Hscope].
    + left. right. exact Hin.
    + apply (add_variable_definition_in_scope tenv x kind y Hwf) in Hscope.
      destruct Hscope as [-> | [_ Hscope]];
        [ left; left; reflexivity | right; exact Hscope ].
Qed.

Lemma handler_env_in_scope (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (tenv_handler : Typing_env.t)
    (y : variable) :
  handler_env tenv params metadata = Some tenv_handler ->
  in_scope_in_terms tenv_handler y ->
  y ∈ params ∨ in_scope_in_terms tenv y.
Proof.
  intros Henv Hy. unfold handler_env in Henv.
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params))
    eqn:Hlength;
    [ | discriminate ].
  apply Nat.eqb_eq in Hlength.
  rewrite <- (map_fst_combine params (hm_params_kinds metadata) Hlength)
    at 1.
  exact (define_variables_checked_in_scope _ tenv tenv_handler y Henv Hy).
Qed.

Lemma bind_params_to_args_in_scope (aliased : simple -> bool)
    (bindings : list (variable * flambda_kind * simple)) :
  ∀ (tenv target : Typing_env.t) (y : variable),
    bind_params_to_args aliased tenv bindings = Some target ->
    y ∈ bound_params bindings ∨ in_scope_in_terms tenv y ->
    in_scope_in_terms target y.
Proof.
  induction bindings as [ | [[x kind] canonical] bindings IH];
    intros tenv target y Hdefined Hy.
  - simpl in Hdefined. injection Hdefined as <-.
    destruct Hy as [[] | Hy]. exact Hy.
  - cbn [bind_params_to_args] in Hdefined.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    apply andb_prop in Htests. destruct Htests as [Hwf _].
    apply (IH _ target y Hdefined).
    assert (Hdefined_scope :
      y = x ∨ in_scope_in_terms tenv y ->
      in_scope_in_terms
        (Typing_env.add_variable_definition tenv x kind Name_mode.normal) y).
    { intros Hcases.
      apply (add_variable_definition_in_scope tenv x kind y Hwf).
      destruct (Pos.eq_dec y x) as [-> | Hne]; [ left; reflexivity | ].
      right. split; [ exact Hne | ].
      destruct Hcases as [-> | Hscope]; [ contradiction | exact Hscope ]. }
    match goal with
    | |- _ ∨ in_scope_in_terms ?tenv1 y =>
      assert (Hstep :
        in_scope_in_terms
          (Typing_env.add_variable_definition tenv x kind Name_mode.normal) y
        -> in_scope_in_terms tenv1 y)
    end.
    { intros Hd.
      destruct (aliased canonical && simple_in_terms_b tenv canonical
                && simple_of_kind_b tenv canonical kind);
        [ | exact Hd ].
      destruct (Typing_env.add_alias
                  (Typing_env.add_variable_definition tenv x kind
                     Name_mode.normal)
                  (Var x) canonical)
        as [[ | result | ] | ] eqn:Hadd; try exact Hd.
      apply (add_alias_in_scope _ _ _ result y Hadd). exact Hd. }
    destruct Hy as [Hin | Hscope].
    + cbn [bound_params map fst] in Hin. destruct Hin as [Heq | Hin].
      * right. apply Hstep, Hdefined_scope. left. symmetry. exact Heq.
      * left. exact Hin.
    + right. apply Hstep, Hdefined_scope. right. exact Hscope.
Qed.

Lemma bound_params_combine (params : list variable) :
  ∀ (kinds : list flambda_kind) (canonicals : list simple),
    length kinds = length params ->
    length canonicals = length params ->
    bound_params (combine (combine params kinds) canonicals) = params.
Proof.
  induction params as [ | x params IH];
    intros [ | kind kinds] [ | canonical canonicals] Hkinds Hcanonicals;
    simpl in Hkinds, Hcanonicals; try discriminate; [ reflexivity | ].
  cbn [combine bound_params map fst]. f_equal. apply IH; congruence.
Qed.

Lemma use_env_in_scope (scope tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (args : list simple) (target : Typing_env.t)
    (y : variable) :
  use_env scope tenv params metadata args = Some target ->
  y ∈ params ∨ in_scope_in_terms tenv y ->
  in_scope_in_terms target y.
Proof.
  unfold use_env. intros Huse Hy.
  destruct (simplify_simples tenv args) as [answers | ]; [ | discriminate ].
  destruct (Nat.eqb (length (hm_params_kinds metadata)) (length params)
            && Nat.eqb (length answers) (length params)) eqn:Hlengths;
    [ | discriminate ].
  apply andb_true_iff in Hlengths. destruct Hlengths as [Hlength Hlength'].
  apply Nat.eqb_eq in Hlength, Hlength'.
  apply (bind_params_to_args_in_scope _ _ tenv target y Huse).
  rewrite (bound_params_combine params (hm_params_kinds metadata)
             (map snd answers) Hlength);
    [ exact Hy | ].
  rewrite length_map. exact Hlength'.
Qed.

Lemma leaf_uses_in_scope (tenv : Typing_env.t) (k : continuation)
    (body : expr) (applications : list (Typing_env.t * list simple))
    (tenv_use : Typing_env.t) (args : list simple) (y : variable) :
  leaf_uses tenv k body = Some applications ->
  (tenv_use, args) ∈ applications ->
  (in_scope_in_terms tenv_use y <-> in_scope_in_terms tenv y).
Proof.
  destruct body
    as [x declared d body | bindings body | k' h body | handlers body | ap
        | ac | scrutinee arms | ];
    simpl; try discriminate.
  - intros [= <-]. destruct (Pos.eqb (ac_continuation ac) k);
      [ | intros [] ].
    intros [[= <- _] | []]. reflexivity.
  - intros [= <-] Hin. apply in_flat_map in Hin.
    destruct Hin as [[d ac] [_ Hin]]. simpl in Hin.
    destruct (Pos.eqb (ac_continuation ac) k); [ | destruct Hin ].
    destruct Hin as [[= <- _] | []]. apply arm_env_in_scope.
Qed.

(* ======================================================================== *)
(* What the body does                                                       *)
(* ======================================================================== *)

(* An [Apply_cont] under the closures of two handlers of [k], defined
   where it is: it applies [k], and the two handlers run under the
   same variables; or it applies another continuation, which is the
   same in the two. [inv] is a predicate that a step keeps: the state
   in which the first handler is entered has it if the state of the
   [Apply_cont] has. *)
Lemma apply_cont_leaf_from (inv : state -> Prop) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr) (ac : apply_cont) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  kept_by_step inv ->
  inv
    (State (Apply_cont ac) vars
       ((k, Closure (Handler params metadata handler) vars conts []) :: conts)
       statics hp rd) ->
  (ac_continuation ac = k ->
   ∀ (vs : list value) (handler_vars : env),
     eval_simples vars (ac_args ac) = Some vs ->
     map kind_of_value vs = hm_params_kinds metadata ->
     bind_params params vs vars = Some handler_vars ->
     inv (State handler handler_vars conts statics hp rd) ->
     refines (State handler handler_vars conts statics hp rd)
       (State handler' handler_vars conts statics hp rd)) ->
  refines
    (State (Apply_cont ac) vars
       ((k, Closure (Handler params metadata handler) vars conts []) :: conts)
       statics hp rd)
    (State (Apply_cont ac) vars
       ((k, Closure (Handler params metadata handler') vars conts [])
        :: conts)
       statics hp rd).
Proof.
  intros Hkept Hinv Hhandler.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs.
  2: { apply step_undefined_refines. rewrite step_apply_cont, Hargs.
       reflexivity. }
  destruct (Pos.eq_dec (ac_continuation ac) k) as [Hk | Hne].
  - destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Hkinds.
    2: { apply step_undefined_refines. rewrite step_apply_cont, Hargs.
         unfold apply_continuation. rewrite Hk, resolve_head. simpl.
         rewrite Hkinds. reflexivity. }
    destruct (bind_params params vs vars) as [handler_vars | ] eqn:Hbind.
    2: { apply step_undefined_refines. rewrite step_apply_cont, Hargs.
         unfold apply_continuation. rewrite Hk, resolve_head. simpl.
         rewrite Hkinds, Hbind. reflexivity. }
    assert (Hstep :
      step
        (State (Apply_cont ac) vars
           ((k, Closure (Handler params metadata handler) vars conts [])
            :: conts)
           statics hp rd)
      = Step (State handler handler_vars conts statics hp rd)).
    { apply (step_apply_cont_handler ac vars _ statics _ rd_ vs params metadata
               handler vars conts [] handler_vars Hargs);
        [ rewrite Hk; apply resolve_head | exact Hkinds | exact Hbind ]. }
    eapply refines_step.
    + exact Hstep.
    + apply (step_apply_cont_handler ac vars _ statics _ rd_ vs params metadata
               handler' vars conts [] handler_vars Hargs);
        [ rewrite Hk; apply resolve_head | exact Hkinds | exact Hbind ].
    + exact (Hhandler Hk vs handler_vars eq_refl
               (proj1 (kinds_eqb_eq _ _) Hkinds) Hbind
               (Hkept _ _ Hinv Hstep)).
  - apply equivalent_refines, step_eq_equivalent.
    rewrite !step_apply_cont, Hargs. unfold apply_continuation.
    rewrite !(resolve_other (ac_continuation ac) k) by exact Hne.
    reflexivity.
Qed.

(* A [Let_cont] whose body is an [Apply_cont] or a [Switch], from the
   states that have a predicate that a step keeps: the handler is
   replaced by one that refines it in the states in which an
   application of the body enters it. Such a state has the predicate,
   and satisfies the environment of the [Let_cont] and the
   environment of the application, with one valuation. *)
Lemma leaf_refines_from (inv : state -> Prop) (tenv : Typing_env.t)
    (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr)
    (applications : list (Typing_env.t * list simple)) :
  kept_by_step inv ->
  leaf_uses tenv k body = Some applications ->
  (∀ (index : nat) (tenv_use : Typing_env.t) (args : list simple)
     (statics : static_env) (hp : heap) (rd : read_record)
       (valuation vars : env)
       (conts : cont_env)
     (vs : list value) (handler_vars : env),
     nth_error applications index = Some (tenv_use, args) ->
     shows tenv statics valuation vars ->
     shows tenv_use statics valuation vars ->
     eval_simples vars args = Some vs ->
     map kind_of_value vs = hm_params_kinds metadata ->
     bind_params params vs vars = Some handler_vars ->
     inv (State handler handler_vars conts statics hp rd) ->
     refines (State handler handler_vars conts statics hp rd)
       (State handler' handler_vars conts statics hp rd)) ->
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    inv
      (State
         (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp rd) ->
    env_satisfies_strictly tenv statics vars ->
    refines
      (State
         (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (Handler params metadata handler') body)
         vars conts statics hp rd).
Proof.
  intros Hkept Hleaf Hhandler vars conts statics hp rd Hinv Hsat.
  destruct (satisfies_shows tenv statics vars Hsat) as [valuation Hshows].
  pose proof (Hkept _ _ Hinv
                (step_let_cont_non_recursive k
                   (Handler params metadata handler) body vars conts
                   statics hp rd))
    as Hinv_body.
  apply let_cont_non_recursive_refines_step.
  destruct body
    as [x declared d body | bindings body | k' h body | handlers body | ap
        | ac | scrutinee arms | ];
    simpl in Hleaf; try discriminate.
  - injection Hleaf as <-.
    apply (apply_cont_leaf_from inv); [ exact Hkept | exact Hinv_body | ].
    intros Hk vs handler_vars Hargs Hkinds Hbind Hinv_handler.
    apply
      (Hhandler 0%nat tenv (ac_args ac) statics hp rd_ valuation vars conts vs
             handler_vars);
      try assumption.
    rewrite (proj2 (Pos.eqb_eq _ _) Hk). reflexivity.
  - injection Hleaf as <-.
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv.
    2: { apply step_undefined_refines. rewrite step_switch, Hv.
         reflexivity. }
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi.
    2: { apply step_undefined_refines. rewrite step_switch, Hv, Hi.
         reflexivity. }
    destruct (find_arm i arms) as [ac | ] eqn:Hfind.
    2: { apply step_undefined_refines. rewrite step_switch, Hv, Hi, Hfind.
         reflexivity. }
    assert (Hstep :
      step
        (State (Switch scrutinee arms) vars
           ((k, Closure (Handler params metadata handler) vars conts [])
            :: conts)
           statics hp rd)
      = Step
          (State (Apply_cont ac) vars
             ((k, Closure (Handler params metadata handler) vars conts [])
              :: conts)
             statics hp rd)).
    { rewrite step_switch, Hv, Hi, Hfind. reflexivity. }
    eapply refines_step;
      [ exact Hstep | rewrite step_switch, Hv, Hi, Hfind; reflexivity | ].
    apply (apply_cont_leaf_from inv);
      [ exact Hkept | exact (Hkept _ _ Hinv_body Hstep) | ].
    intros Hk vs handler_vars Hargs Hkinds Hbind Hinv_handler.
    assert (Hvalue : v = Value_const (Naked_immediate i)).
    { unfold naked_immediate_of_value in Hi.
      destruct v as [c | | | | | | | | ]; try discriminate Hi.
      destruct c; try discriminate Hi. injection Hi as <-. reflexivity. }
    assert (Hin :
      (arm_env tenv scrutinee i, ac_args ac)
      ∈ flat_map
          (fun arm =>
             if Pos.eqb (ac_continuation (snd arm)) k
             then [(arm_env tenv scrutinee (fst arm), ac_args (snd arm))]
             else [])
          arms).
    { apply in_flat_map. exists (i, ac).
      split; [ exact (find_arm_in i arms ac Hfind) | ].
      cbn [fst snd]. rewrite (proj2 (Pos.eqb_eq _ _) Hk). left.
      reflexivity. }
    apply In_nth_error in Hin. destruct Hin as [index Hnth].
    apply (Hhandler index (arm_env tenv scrutinee i) (ac_args ac) statics
             hp rd_ valuation vars conts vs handler_vars Hnth Hshows);
      try assumption.
    apply arm_env_shows; [ exact Hshows | ]. rewrite Hv, Hvalue. reflexivity.
Qed.

(* The same from every state. *)
Lemma leaf_refines (tenv : Typing_env.t) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr)
    (applications : list (Typing_env.t * list simple)) :
  leaf_uses tenv k body = Some applications ->
  (∀ (index : nat) (tenv_use : Typing_env.t) (args : list simple)
     (statics : static_env) (hp : heap) (rd : read_record)
       (valuation vars : env)
       (conts : cont_env)
     (vs : list value) (handler_vars : env),
     nth_error applications index = Some (tenv_use, args) ->
     shows tenv statics valuation vars ->
     shows tenv_use statics valuation vars ->
     eval_simples vars args = Some vs ->
     map kind_of_value vs = hm_params_kinds metadata ->
     bind_params params vs vars = Some handler_vars ->
     refines (State handler handler_vars conts statics hp rd)
       (State handler' handler_vars conts statics hp rd)) ->
  expr_refines_under tenv
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  intros Hleaf Hhandler vars conts statics hp rd Hsat.
  apply (leaf_refines_from (fun _ => True) tenv k params metadata handler
           handler' body applications);
    [ intros s s' _ _; exact I | exact Hleaf | | exact I | exact Hsat ].
  intros index tenv_use args statics' hp' rd' valuation vars' conts' vs
    handler_vars Hnth Hshows Hshows_use Hargs Hkinds Hbind _.
  exact (Hhandler index tenv_use args statics' hp' rd_ valuation vars' conts' vs
           handler_vars Hnth Hshows Hshows_use Hargs Hkinds Hbind).
Qed.

(* One application, from the states that have a predicate that a step
   keeps: the form that the function which prunes uses. *)
Lemma let_cont_handler_at_use_from (inv : state -> Prop)
    (scope tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (tenv_use : Typing_env.t) (args : list simple) (target : Typing_env.t) :
  kept_by_step inv ->
  leaf_uses tenv k body = Some [(tenv_use, args)] ->
  use_env scope tenv_use params metadata args = Some target ->
  (∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
     (rd : read_record),
     inv (State handler vars conts statics hp rd) ->
     env_satisfies_strictly target statics vars ->
     refines (State handler vars conts statics hp rd)
       (State handler' vars conts statics hp rd)) ->
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    inv
      (State
         (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp rd) ->
    env_satisfies_strictly tenv statics vars ->
    refines
      (State
         (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (Handler params metadata handler') body)
         vars conts statics hp rd).
Proof.
  intros Hkept Hleaf Huse Hhandler.
  apply (leaf_refines_from inv tenv k params metadata handler handler' body
           [(tenv_use, args)] Hkept Hleaf).
  intros index tenv_use' args' statics hp rd valuation vars conts vs
    handler_vars
    Hnth _ Hshows Hargs Hkinds Hbind Hinv.
  destruct index as [ | index];
    [ | simpl in Hnth; destruct index; discriminate Hnth ].
  injection Hnth as <- <-. apply Hhandler; [ exact Hinv | ].
  destruct (use_env_shows scope tenv_use params metadata args target statics
              valuation vars vs handler_vars Huse Hshows Hargs Hkinds Hbind)
    as [valuation' [_ Hshows']].
  exact (shows_satisfies target statics valuation' handler_vars Hshows').
Qed.

(* ======================================================================== *)
(* One application                                                          *)
(* ======================================================================== *)

Theorem let_cont_handler_at_use : let_cont_handler_at_use_statement.
Proof.
  intros scope tenv k params metadata handler handler' body tenv_use args
    target Hleaf Huse Hhandler.
  apply (leaf_refines tenv k params metadata handler handler' body
           [(tenv_use, args)] Hleaf).
  intros index tenv_use' args' statics hp rd valuation vars conts vs
    handler_vars
    Hnth _ Hshows Hargs Hkinds Hbind.
  destruct index as [ | index];
    [ | simpl in Hnth; destruct index; discriminate Hnth ].
  injection Hnth as <- <-. apply Hhandler.
  destruct (use_env_shows scope tenv_use params metadata args target statics
              valuation vars vs handler_vars Huse Hshows Hargs Hkinds Hbind)
    as [valuation' [_ Hshows']].
  exact (shows_satisfies target statics valuation' handler_vars Hshows').
Qed.

(* ======================================================================== *)
(* The join of the applications                                             *)
(* ======================================================================== *)

Lemma forall2_nth_left {X Y : Type} (R : X -> Y -> Prop) (l : list X) :
  ∀ (l' : list Y) (n : nat) (x : X),
    Forall2 R l l' ->
    nth_error l n = Some x ->
    ∃ y : Y, nth_error l' n = Some y ∧ R x y.
Proof.
  induction l as [ | x0 l IH]; intros l' n x Hall Hnth;
    [ destruct n; discriminate | ].
  inversion Hall as [ | x1 y0 l1 l1' Hhead Htail]; subst.
  destruct n as [ | n].
  - injection Hnth as <-. exists y0. split; [ reflexivity | exact Hhead ].
  - exact (IH l1' n x Htail Hnth).
Qed.

Lemma forall2_nth_right {X Y : Type} (R : X -> Y -> Prop) (l : list X) :
  ∀ (l' : list Y) (n : nat) (y : Y),
    Forall2 R l l' ->
    nth_error l' n = Some y ->
    ∃ x : X, nth_error l n = Some x ∧ R x y.
Proof.
  induction l as [ | x0 l IH]; intros l' n y Hall Hnth.
  - inversion Hall; subst. destruct n; discriminate.
  - inversion Hall as [ | x1 y0 l1 l1' Hhead Htail]; subst.
    destruct n as [ | n].
    + injection Hnth as <-. exists x0. split; [ reflexivity | exact Hhead ].
    + exact (IH l1' n y Htail Hnth).
Qed.

Theorem let_cont_handler_joined : let_cont_handler_joined_statement.
Proof.
  intros well_formed A avoid scope tenv k params metadata handler handler'
    body applications inputs target analysis Hleaf Hsource Huses Hguarantees
    Hhandler.
  apply (leaf_refines tenv k params metadata handler handler' body
           applications Hleaf).
  intros index tenv_use args statics hp rd valuation vars conts vs handler_vars
    Hnth Hshows Hshows_use Hargs Hkinds Hbind.
  apply Hhandler.
  destruct (forall2_nth_left _ _ _ index _ Huses Hnth)
    as [[id use] [Hnth' Huse]].
  cbn [fst snd] in Huse.
  apply (join_guarantees_in_machine well_formed A avoid inputs target
           analysis index use statics handler_vars Hguarantees).
  - intros index' use' x [id' Hnth_use] Hx.
    destruct (forall2_nth_right _ _ _ index' _ Huses Hnth_use)
      as [[tenv_use' args'] [Hnth_app Huse']].
    cbn [fst snd] in Huse'.
    apply (use_env_in_scope scope tenv_use' params metadata args' use' x
             Huse').
    destruct (handler_env_in_scope tenv params metadata (source inputs) x
                Hsource Hx)
      as [Hin | Hscope]; [ left; exact Hin | right ].
    apply (leaf_uses_in_scope tenv k body applications tenv_use' args' x
             Hleaf (nth_error_In _ _ Hnth_app)).
    exact Hscope.
  - exists id. exact Hnth'.
  - destruct (use_env_shows scope tenv_use params metadata args use statics
                valuation vars vs handler_vars Huse Hshows_use Hargs Hkinds
                Hbind)
      as [valuation' [Hbind' [Hbound' [Hsat' Hagrees']]]].
    destruct (handler_env_shows tenv params metadata (source inputs) statics
                valuation vars vs handler_vars Hsource Hshows Hkinds Hbind)
      as [valuation'' [Hbind'' [_ [Hsat'' _]]]].
    assert (valuation'' = valuation') as -> by congruence.
    split; [ exact Hbound' | ]. exists valuation'.
    split; [ exact Hsat' | split; [ exact Hsat'' | exact Hagrees' ] ].
Qed.
