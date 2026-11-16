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


(* Proofs of the statements of FlambdaMeetScopesSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs
  FlambdaMeetStepsSpec FlambdaMeetStepsProofs FlambdaMeetEnvSpec
  FlambdaMeetEnvProofs FlambdaMeetScopesSpec.
Import ListNotations.

(* The proofs below, as far as the end of the section, are for any
   predicate on static environments that implies validity
   (FlambdaMeetValidity.v): in them the names of the four contracts
   that suppose a valid static environment stand for the same with
   that predicate. The theorems of the signed statements are after the
   section, at validity itself. *)
From OxCamlFlambda2Types Require Import FlambdaMeetValidity.

Section Validity.

Context {V : validity}.

Abbreviation statics_valid := valid.
Abbreviation meet_expanded_head_sound := meet_expanded_head_sound_for.
Abbreviation join_scoped_envs_sound := join_scoped_envs_sound_for.
Abbreviation meet_call_ok := meet_call_ok_for.
Abbreviation part_meet_ok := part_meet_ok_for.
Abbreviation meet_disjunction_right := meet_disjunction_right_for.
Abbreviation type_part_meet_ok := type_part_meet_ok_for.
Abbreviation meet_row_like_for_closures_right :=
  meet_row_like_for_closures_right_for.
Abbreviation value_heads_right := value_heads_right_for.

(* ======================================================================== *)
(* Valuations                                                               *)
(* ======================================================================== *)

(* A valuation that agrees with one that satisfies an environment, on
   the variables that the environment knows, satisfies it. *)
Lemma satisfied_strictly_extended (statics : static_env)
    (tenv : Typing_env.t) (valuation valuation' : env) :
  typing_env_well_formed tenv = true ->
  extended_from tenv valuation valuation' ->
  typing_env_satisfied_strictly statics valuation tenv ->
  typing_env_satisfied_strictly statics valuation' tenv.
Proof.
  intros Hwf Hextended.
  apply (satisfied_but_coincide (fun _ => False) statics valuation valuation'
           tenv).
  intros x Hmentions. symmetry. apply Hextended.
  destruct (Typing_env.mem None tenv (Name_var x)) eqn:Hmem; [ reflexivity | ].
  exfalso. exact (well_formed_not_mentioned tenv x Hwf Hmem Hmentions).
Qed.

(* A name that has an equation in a level has a type. *)
Lemma equation_name_bound (tenv : Typing_env.t)
    (one_level : Typing_env.One_level.t) (n : name) (ty : flambda_type) :
  typing_env_invariant tenv ->
  one_level ∈ Typing_env.current_level tenv :: Typing_env.prev_levels tenv ->
  (n, ty)
  ∈ Typing_env_level.equations (Typing_env.One_level.level one_level) ->
  Typing_env.mem None tenv n = true.
Proof.
  intros Hinvariant Hin Hequation.
  assert (Hall : (n, ty) ∈ all_equations tenv).
  { unfold all_equations, all_levels. apply in_flat_map.
    exists (Typing_env.One_level.level one_level).
    split; [ apply in_map; exact Hin | exact Hequation ]. }
  destruct (Keyed.lookup name_eqb n (all_equations tenv)) as [ ty' | ]
    eqn:Hlookup.
  - destruct (proj2 Hinvariant n ty' Hlookup) as [binding_time_and_mode Hfound].
    apply (simple_of_kind_mem tenv n (kind ty') Hinvariant).
    apply simple_of_kind_name. left. exists ty', binding_time_and_mode.
    split; [ exact Hfound | reflexivity ].
  - exfalso. clear Hin Hequation. revert Hlookup Hall.
    generalize (all_equations tenv) as equations.
    induction equations as [ | [m ty0] equations IH ]; intros Hlookup Hall;
      [ contradiction | ].
    cbn in Hlookup. destruct (name_eqb n m) eqn:Hsame;
      [ discriminate Hlookup | ].
    destruct Hall as [ Heq | Hall ]; [ | exact (IH Hlookup Hall) ].
    injection Heq as -> _.
    rewrite (proj2 (name_eqb_eq _ _) eq_refl) in Hsame. discriminate Hsame.
Qed.

(* The equations of the levels are on names that have a value under a
   valuation that agrees with one under which they have. *)
Lemma equations_valued_extended (tenv : Typing_env.t) (cut_after : Scope.t)
    (valuation valuation' : env) :
  typing_env_invariant tenv ->
  extended_from tenv valuation valuation' ->
  equations_valued tenv cut_after valuation ->
  equations_valued tenv cut_after valuation'.
Proof.
  intros Hinvariant Hextended Hvalued one_level n ty Hin Hscope Hequation.
  destruct (Hvalued one_level n ty Hin Hscope Hequation) as [v Hv].
  exists v. rewrite <- Hv.
  apply (eval_simple_extended tenv valuation valuation' (simple_of_name n)
           Hextended).
  rewrite mem_simple_of_name.
  exact (equation_name_bound tenv one_level n ty Hinvariant Hin Hequation).
Qed.

(* A valuation with some of its variables left out. *)
Definition restrict_valuation (keep : variable -> bool) (valuation : env)
    : env :=
  filter (fun binding => keep (fst binding)) valuation.

Lemma lookup_restrict_valuation (keep : variable -> bool) (valuation : env)
    (x : variable) :
  lookup x (restrict_valuation keep valuation)
  = if keep x then lookup x valuation else None.
Proof.
  induction valuation as [ | [y v] valuation IH ].
  - destruct (keep x); reflexivity.
  - cbn [restrict_valuation filter fst].
    fold (restrict_valuation keep valuation).
    destruct (Pos.eqb x y) eqn:Hsame.
    + apply Pos.eqb_eq in Hsame. subst y.
      destruct (keep x) eqn:Hkeep.
      * cbn [lookup]. rewrite Pos.eqb_refl. reflexivity.
      * rewrite IH. try rewrite Hkeep. reflexivity.
    + destruct (keep y).
      * cbn [lookup]. rewrite Hsame. exact IH.
      * rewrite IH. cbn [lookup]. rewrite Hsame. reflexivity.
Qed.

(* ======================================================================== *)
(* One scope further                                                        *)
(* ======================================================================== *)

Lemma increment_scope_fits (tenv : Typing_env.t) :
  typing_env_fits tenv -> typing_env_fits (Typing_env.increment_scope tenv).
Proof.
  intros (Hinvariant & Hwf & Hkinded & Hequations & Hbottom).
  split; [ exact (increment_scope_invariant tenv Hinvariant) | ].
  split; [ exact (increment_scope_types_well_formed tenv Hwf) | ].
  split; [ exact (increment_scope_kinded tenv Hkinded) | ].
  split; [ exact (increment_scope_equations_kinded tenv Hequations) | ].
  exact Hbottom.
Qed.

(* A level of an environment is not after its current scope. *)
Lemma level_scope_bounded (tenv : Typing_env.t)
    (one_level : Typing_env.One_level.t) :
  typing_env_invariant tenv ->
  one_level ∈ Typing_env.current_level tenv :: Typing_env.prev_levels tenv ->
  (Typing_env.One_level.scope one_level <= Typing_env.current_scope tenv)%Z.
Proof.
  intros [Hwf _] Hin. apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as (_ & (_ & _ & _ & _ & _ & Hscopes & _) & _).
  unfold all_scopes in Hscopes. unfold Typing_env.current_scope.
  revert Hin Hscopes.
  generalize (Typing_env.current_level tenv) as current.
  induction (Typing_env.prev_levels tenv) as [ | previous levels IH ];
    intros current Hin Hscopes.
  - destruct Hin as [ <- | [] ]. lia.
  - destruct Hin as [ <- | Hin ]; [ lia | ].
    cbn [map scopes_decreasing] in Hscopes.
    destruct Hscopes as [Hless Hscopes].
    specialize (IH previous Hin Hscopes). lia.
Qed.

(* No level is after the scope before the one entered. *)
Lemma increment_scope_valued (tenv : Typing_env.t) (valuation : env) :
  typing_env_invariant tenv ->
  equations_valued (Typing_env.increment_scope tenv)
    (Typing_env.current_scope tenv) valuation.
Proof.
  intros Hinvariant one_level n ty Hin Hscope Hequation.
  destruct Hin as [ <- | Hin ]; [ destruct Hequation | ].
  exfalso.
  assert (Hbound := level_scope_bounded tenv one_level Hinvariant Hin).
  lia.
Qed.

(* ======================================================================== *)
(* From one environment to another that knows the same at the same kinds    *)
(* ======================================================================== *)

Lemma knows_at_kinds_trans (tenv1 tenv2 tenv3 : Typing_env.t) :
  knows_at_kinds tenv1 tenv2 -> knows_at_kinds tenv2 tenv3 ->
  knows_at_kinds tenv1 tenv3.
Proof.
  intros [Hmem12 Hkind12] [Hmem23 Hkind23]. split.
  - intros n Hn. exact (Hmem23 n (Hmem12 n Hn)).
  - intros s kind Hs. exact (Hkind23 s kind (Hkind12 s kind Hs)).
Qed.

Lemma tenv_extends_knows (tenv tenv' : Typing_env.t) :
  tenv_extends tenv tenv' -> knows_at_kinds tenv tenv'.
Proof.
  intros Hextends. split.
  - intros n Hn. exact (tenv_extends_mem _ _ n Hextends Hn).
  - exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends))))).
Qed.

Lemma meet_env_extends_knows (menv menv' : Meet_env.t) :
  meet_env_extends menv menv' ->
  knows_at_kinds (Meet_env.typing_env menv) (Meet_env.typing_env menv').
Proof.
  intros Hextends.
  exact (tenv_extends_knows _ _ (meet_env_extends_tenv _ _ Hextends)).
Qed.

Lemma type_fits_by_kinds : fits_by_kinds type_fits.
Proof.
  intros tenv tenv' ty [Hmem Hkind].
  apply type_fits_kept; [ | exact Hkind ].
  intros x Hx. exact (Hmem (Name_var x) Hx).
Qed.

Lemma extension_fits_kept (tenv tenv' : Typing_env.t)
    (extension : env_extension) :
  knows_at_kinds tenv tenv' ->
  extension_fits tenv extension -> extension_fits tenv' extension.
Proof.
  intros Hknows (Hwf & Hfits & Hkinded). destruct extension as [equations].
  split; [ exact Hwf | ]. split.
  - intros n ty Hlookup. destruct (Hfits n ty Hlookup) as [Hmem Hty].
    split; [ exact (proj1 Hknows n Hmem) | ].
    exact (type_fits_by_kinds tenv tenv' ty Hknows Hty).
  - intros n ty Hlookup. exact (proj2 Hknows _ _ (Hkinded n ty Hlookup)).
Qed.

Lemma optional_extension_fits_kept (tenv tenv' : Typing_env.t)
    (extension : option env_extension) :
  knows_at_kinds tenv tenv' ->
  optional_extension_fits tenv extension ->
  optional_extension_fits tenv' extension.
Proof.
  intros Hknows. destruct extension as [ extension | ]; [ | intros H; exact H ].
  exact (extension_fits_kept tenv tenv' extension Hknows).
Qed.

(* An extension that fits is true under a valuation that agrees, on
   what the environment knows, with one under which it is. *)
Lemma env_extension_satisfied_extended (statics : static_env)
    (tenv : Typing_env.t) (valuation valuation' : env)
    (extension : env_extension) :
  env_extension_fits tenv extension ->
  extended_from tenv valuation valuation' ->
  env_extension_satisfied statics valuation extension ->
  env_extension_satisfied statics valuation' extension.
Proof.
  destruct extension as [equations]. intros Hfits Hextended Hsatisfied.
  apply env_extension_lookup. intros n ty Hlookup.
  destruct (proj1 (env_extension_lookup statics valuation equations)
              Hsatisfied n ty Hlookup) as (v & Hv & Hty).
  destruct (Hfits n ty Hlookup) as [Hmem Hty_fits].
  exists v. split.
  - rewrite <- Hv.
    apply (eval_simple_extended tenv valuation valuation' (simple_of_name n)
             Hextended).
    rewrite mem_simple_of_name. exact Hmem.
  - exact (type_satisfied_extended statics tenv valuation valuation' v ty
             Hty_fits Hextended Hty).
Qed.

Lemma optional_extension_satisfied_extended (statics : static_env)
    (tenv : Typing_env.t) (valuation valuation' : env)
    (extension : option env_extension) :
  optional_extension_fits tenv extension ->
  extended_from tenv valuation valuation' ->
  optional_extension_satisfied statics valuation extension ->
  optional_extension_satisfied statics valuation' extension.
Proof.
  destruct extension as [ extension | ]; [ | intros _ _ H; exact H ].
  intros (_ & Hfits & _).
  exact (env_extension_satisfied_extended statics tenv valuation valuation'
           extension Hfits).
Qed.

(* A valuation that agrees with another on what an environment knows
   agrees with it on what an environment that knows less knows. *)
Lemma extended_from_knows (tenv tenv' : Typing_env.t)
    (valuation valuation' : env) :
  knows_at_kinds tenv tenv' ->
  extended_from tenv' valuation valuation' ->
  extended_from tenv valuation valuation'.
Proof.
  intros [Hmem _] Hextended x Hx. exact (Hextended x (Hmem _ Hx)).
Qed.

(* ======================================================================== *)
(* Defining the variables of a level                                        *)
(* ======================================================================== *)

Section Define.

Variable choices : meet_choices.
Hypothesis Hchoice : define_again choices = false.

(* A variable that the environment knows is left; one that it does not
   know is defined. *)
Lemma define_in_types_cases (menv menv' : Meet_env.t) (x : variable)
    (k : flambda_kind) :
  define_in_types choices menv x k = Some menv' ->
  (Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = true
   ∧ menv' = menv)
  ∨ (Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = false
     ∧ menv' = Meet_env.add_variable_definition menv x k Name_mode.in_types).
Proof.
  unfold define_in_types. rewrite Hchoice. cbn [orb].
  destruct (Typing_env.mem None (Meet_env.typing_env menv) (Name_var x));
    cbn [negb].
  - intros [= <-]. left. split; reflexivity.
  - intros Hdefined. right. split; [ reflexivity | ].
    unfold Meet_env.add_definition, Typing_env.add_definition in Hdefined.
    cbv beta iota in Hdefined. injection Hdefined as <-. reflexivity.
Qed.

(* What defining the variables of [vars] that have a kind does to the
   environment of a meet. *)
Definition defined_from (kind_of : variable -> option flambda_kind)
    (vars : list variable) (menv menv' : Meet_env.t) : Prop :=
  let tenv := Meet_env.typing_env menv in
  let tenv' := Meet_env.typing_env menv' in
  meet_env_extends menv menv'
  ∧ typing_env_fits tenv'
  ∧ Meet_env.names_to_avoid menv' = Meet_env.names_to_avoid menv
  ∧ (∀ y : variable,
       Typing_env.mem None tenv' (Name_var y) = true ->
       Typing_env.mem None tenv (Name_var y) = true
       ∨ (y ∈ vars ∧ kind_of y <> None))
  ∧ (∀ (y : variable) (k : flambda_kind),
       y ∈ vars ->
       kind_of y = Some k ->
       Typing_env.mem None tenv (Name_var y) = false ->
       simple_of_kind tenv' (Var y) k)
  ∧ (∀ (y : variable) (k : flambda_kind),
       y ∈ vars ->
       kind_of y = Some k ->
       Typing_env.mem None tenv' (Name_var y) = true)
  ∧ (∀ (statics : static_env) (valuation : env),
       typing_env_satisfied_strictly statics valuation tenv ->
       (∀ (y : variable) (k : flambda_kind) (v : value),
          y ∈ vars ->
          kind_of y = Some k ->
          Typing_env.mem None tenv (Name_var y) = false ->
          lookup y valuation = Some v ->
          kind_of_value v = k) ->
       typing_env_satisfied_strictly statics valuation tenv')
  ∧ (∀ valuation : env, keeps_equations_valued tenv tenv' valuation valuation).

Lemma defined_from_nil (kind_of : variable -> option flambda_kind)
    (menv : Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  defined_from kind_of [] menv menv.
Proof.
  intros Hfits.
  split; [ apply meet_env_extends_refl | ]. split; [ exact Hfits | ].
  split; [ reflexivity | ]. split; [ intros y Hmem; left; exact Hmem | ].
  split; [ intros y k [] | ]. split; [ intros y k [] | ].
  split; [ intros statics valuation Hsatisfied _; exact Hsatisfied | ].
  intros valuation. apply keeps_equations_valued_refl.
Qed.

(* A variable that is known, or has no kind, before the others. *)
Lemma defined_from_skip (kind_of : variable -> option flambda_kind)
    (x : variable) (vars : list variable) (menv menv' : Meet_env.t) :
  Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = true
  ∨ kind_of x = None ->
  defined_from kind_of vars menv menv' ->
  defined_from kind_of (x :: vars) menv menv'.
Proof.
  intros Hx (Hextends & Hfits' & Hnames & Hmem & Hkind & Hknown & Hsatisfied
             & Hvalued).
  split; [ exact Hextends | ]. split; [ exact Hfits' | ].
  split; [ exact Hnames | ]. split.
  { intros y Hy. destruct (Hmem y Hy) as [ Hold | [Hin Hsome] ];
      [ left; exact Hold | right ].
    split; [ right; exact Hin | exact Hsome ]. }
  split.
  { intros y k [ <- | Hin ] Hk Hunknown;
      [ | exact (Hkind y k Hin Hk Hunknown) ].
    destruct Hx as [ Hx | Hx ]; [ rewrite Hx in Hunknown | rewrite Hx in Hk ];
      discriminate. }
  split.
  { intros y k [ <- | Hin ] Hk; [ | exact (Hknown y k Hin Hk) ].
    destruct Hx as [ Hx | Hx ]; [ | rewrite Hx in Hk; discriminate Hk ].
    exact (tenv_extends_mem _ _ _ (meet_env_extends_tenv _ _ Hextends) Hx). }
  split; [ | exact Hvalued ].
  intros statics valuation Hs Hkinds. apply (Hsatisfied statics valuation Hs).
  intros y k v Hin. exact (Hkinds y k v (or_intror Hin)).
Qed.

(* A variable that is not known is defined. *)
Lemma define_fresh (menv : Meet_env.t) (x : variable) (k : flambda_kind) :
  typing_env_fits (Meet_env.typing_env menv) ->
  Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = false ->
  let menv' := Meet_env.add_variable_definition menv x k Name_mode.in_types in
  let tenv := Meet_env.typing_env menv in
  let tenv' := Meet_env.typing_env menv' in
  meet_env_kept menv menv'
  ∧ simple_of_kind tenv' (Var x) k
  ∧ (∀ y : variable,
       Typing_env.mem None tenv' (Name_var y) = true ->
       y = x ∨ Typing_env.mem None tenv (Name_var y) = true)
  ∧ (∀ (statics : static_env) (valuation : env),
       typing_env_satisfied_strictly statics valuation tenv ->
       (∀ v : value, lookup x valuation = Some v -> kind_of_value v = k) ->
       typing_env_satisfied_strictly statics valuation tenv')
  ∧ (∀ valuation : env, keeps_equations_valued tenv tenv' valuation valuation).
Proof.
  intros Hfits Hfresh menv' tenv tenv'.
  change tenv'
    with (Typing_env.add_variable_definition tenv x k Name_mode.in_types).
  split.
  { exact (add_variable_definition_keeps_for menv x k Name_mode.in_types Hfits
             Hfresh). }
  split.
  { apply (simple_of_kind_name _ (Name_var x)). left.
    eexists. eexists. split.
    - rewrite add_variable_definition_names.
      apply (keyed_lookup_head name_eqb name_eqb_eq).
    - apply kind_unknown. }
  split.
  { intros y Hy. rewrite mem_var_is_bound in Hy.
    rewrite add_variable_definition_names in Hy.
    unfold Keyed.is_bound in Hy. cbn [Keyed.lookup] in Hy.
    destruct (name_eqb (Name_var y) (Name_var x)) eqn:Hsame.
    - apply name_eqb_eq in Hsame. injection Hsame as ->. left. reflexivity.
    - right. rewrite mem_var_is_bound. exact Hy. }
  split.
  { intros statics valuation Hsatisfied Hkind.
    exact (add_variable_definition_satisfied (fun _ => False) statics
             valuation tenv x k Name_mode.in_types Hsatisfied Hkind). }
  intros valuation cut_after Hvalued one_level n ty Hin Hscope Hequation.
  destruct Hin as [ <- | Hin ].
  - exact (Hvalued (Typing_env.current_level tenv) n ty (or_introl eq_refl)
             Hscope Hequation).
  - exact (Hvalued one_level n ty (or_intror Hin) Hscope Hequation).
Qed.

(* The variables of a list, each defined with the kind that it has. *)
Definition define_all (kind_of : variable -> option flambda_kind)
    (vars : list variable) (menv : option Meet_env.t) : option Meet_env.t :=
  List.fold_left
    (fun acc x =>
       match kind_of x with
       | Some kind =>
         let* env := acc in
         define_in_types choices env x kind
       | None => acc
       end)
    vars menv.

Lemma define_all_none (kind_of : variable -> option flambda_kind)
    (vars : list variable) :
  define_all kind_of vars None = None.
Proof.
  induction vars as [ | x vars IH ]; [ reflexivity | ].
  cbn [define_all List.fold_left]. destruct (kind_of x); exact IH.
Qed.

Lemma define_all_defined (kind_of : variable -> option flambda_kind)
    (vars : list variable) :
  ∀ menv menv' : Meet_env.t,
    typing_env_fits (Meet_env.typing_env menv) ->
    define_all kind_of vars (Some menv) = Some menv' ->
    defined_from kind_of vars menv menv'.
Proof.
  induction vars as [ | x vars IH ]; intros menv menv' Hfits Hdefine.
  { injection Hdefine as <-. exact (defined_from_nil kind_of menv Hfits). }
  cbn [define_all List.fold_left] in Hdefine.
  destruct (kind_of x) as [ k | ] eqn:Hk.
  2:{ apply defined_from_skip; [ right; exact Hk | ].
      exact (IH menv menv' Hfits Hdefine). }
  cbv beta iota in Hdefine.
  destruct (define_in_types choices menv x k) as [ menv1 | ] eqn:Hdefined.
  2:{ change (define_all kind_of vars None = Some menv') in Hdefine.
      rewrite define_all_none in Hdefine. discriminate Hdefine. }
  change (define_all kind_of vars (Some menv1) = Some menv') in Hdefine.
  destruct (define_in_types_cases menv menv1 x k Hdefined)
    as [ [Hknown ->] | [Hfresh ->] ].
  { apply defined_from_skip; [ left; exact Hknown | ].
    exact (IH menv menv' Hfits Hdefine). }
  destruct (define_fresh menv x k Hfits Hfresh)
    as ([Hextends1 Hfits1] & Hkind1 & Hmem1 & Hsatisfied1 & Hvalued1).
  destruct (IH _ menv' Hfits1 Hdefine)
    as (Hextends & Hfits' & Hnames & Hmem & Hkind & Hknown & Hsatisfied
        & Hvalued).
  assert (Htenv1 := meet_env_extends_tenv _ _ Hextends1).
  assert (Htenv := meet_env_extends_tenv _ _ Hextends).
  split; [ exact (meet_env_extends_trans _ _ _ Hextends1 Hextends) | ].
  split; [ exact Hfits' | ]. split; [ exact Hnames | ]. split.
  { intros y Hy. destruct (Hmem y Hy) as [ Hold | [Hin Hsome] ].
    - destruct (Hmem1 y Hold) as [ -> | Hold' ]; [ right | left; exact Hold' ].
      split; [ left; reflexivity | rewrite Hk; discriminate ].
    - right. split; [ right; exact Hin | exact Hsome ]. }
  split.
  { intros y k' Hin Hk' Hunknown. destruct (Pos.eq_dec y x) as [ -> | Hne ].
    - rewrite Hk in Hk'. injection Hk' as <-.
      exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv)))) _ _ Hkind1).
    - destruct Hin as [ Heq | Hin ]; [ symmetry in Heq; contradiction | ].
      apply (Hkind y k' Hin Hk').
      destruct (Typing_env.mem None
                  (Meet_env.typing_env
                     (Meet_env.add_variable_definition menv x k
                        Name_mode.in_types))
                  (Name_var y)) eqn:Hy; [ | reflexivity ].
      destruct (Hmem1 y Hy) as [ -> | Hold ]; [ contradiction | ].
      rewrite Hold in Hunknown. discriminate Hunknown. }
  split.
  { intros y k' [ <- | Hin ] Hk'; [ | exact (Hknown y k' Hin Hk') ].
    apply (tenv_extends_mem _ _ _ Htenv).
    exact (simple_of_kind_mem _ (Name_var x) k (proj1 Hfits1) Hkind1). }
  split.
  { intros statics valuation Hs Hkinds.
    apply (Hsatisfied statics valuation).
    - apply (Hsatisfied1 statics valuation Hs). intros v Hv.
      exact (Hkinds x k v (or_introl eq_refl) Hk Hfresh Hv).
    - intros y k' v Hin Hk' Hunknown.
      apply (Hkinds y k' v (or_intror Hin) Hk').
      destruct (Typing_env.mem None (Meet_env.typing_env menv) (Name_var y))
        eqn:Hy; [ | reflexivity ].
      rewrite (tenv_extends_mem _ _ _ Htenv1 Hy) in Hunknown.
      discriminate Hunknown. }
  intros valuation.
  exact (keeps_equations_valued_trans _ _ _ _ _ _ (Hvalued1 valuation)
           (Hvalued valuation)).
Qed.

(* A variable that the environment does not know may not occur in
   terms there. *)
Lemma unknown_not_in_terms (tenv : Typing_env.t) (y : variable) :
  Typing_env.mem None tenv (Name_var y) = false ->
  Typing_env.mem (Some Name_mode.normal) tenv (Name_var y) = false.
Proof.
  unfold Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var y) (Typing_env.names_to_types tenv))
    as [[ty binding_time_and_mode] | ]; [ discriminate | reflexivity ].
Qed.

(* Defining keeps a variable out of terms: a variable is defined in
   the name mode of the variables that exist only in types, and the
   others are as they were. This is so whatever [define_again] is. *)
Lemma define_in_types_not_in_terms (menv menv' : Meet_env.t) (x : variable)
    (k : flambda_kind) (y : variable) :
  define_in_types choices menv x k = Some menv' ->
  Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv)
    (Name_var y)
  = false ->
  Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv')
    (Name_var y)
  = false.
Proof.
  unfold define_in_types.
  destruct (define_again choices
            || negb (Typing_env.mem None (Meet_env.typing_env menv)
                       (Name_var x)));
    [ | intros [= <-] Hy; exact Hy ].
  unfold Meet_env.add_definition, Typing_env.add_definition.
  cbv beta iota. intros [= <-] Hy.
  cbn [Meet_env.typing_env Meet_env.with_typing_env].
  unfold Typing_env.mem in *. rewrite add_variable_definition_names.
  rewrite keyed_lookup_cons.
  destruct (name_eqb (Name_var y) (Name_var x)); [ | exact Hy ].
  unfold Binding_time.With_name_mode.scoped_name_mode. cbn.
  destruct (_ || _); reflexivity.
Qed.

Lemma define_all_not_in_terms (kind_of : variable -> option flambda_kind)
    (vars : list variable) (y : variable) :
  ∀ menv menv' : Meet_env.t,
    define_all kind_of vars (Some menv) = Some menv' ->
    Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv)
      (Name_var y)
    = false ->
    Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv')
      (Name_var y)
    = false.
Proof.
  induction vars as [ | x vars IH ]; intros menv menv' Hall Hy;
    cbn [define_all List.fold_left] in Hall.
  - injection Hall as <-. exact Hy.
  - destruct (kind_of x) as [ k | ]; [ | exact (IH menv menv' Hall Hy) ].
    cbv beta iota in Hall.
    destruct (define_in_types choices menv x k) as [ menv1 | ] eqn:Hdefined.
    + exact (IH menv1 menv' Hall
               (define_in_types_not_in_terms menv menv1 x k y Hdefined Hy)).
    + change (define_all kind_of vars None = Some menv') in Hall.
      rewrite define_all_none in Hall. discriminate Hall.
Qed.

(* [add_defined_vars] keeps a variable out of terms; in particular the
   variables that it defines may not occur in terms. *)
Lemma add_defined_vars_not_in_terms (menv menv' : Meet_env.t)
    (level : Typing_env_level.t) (y : variable) :
  level_indexed level ->
  add_defined_vars choices menv level = Some menv' ->
  Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv)
    (Name_var y)
  = false ->
  Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv')
    (Name_var y)
  = false.
Proof.
  intros Hindexed Hadd. unfold add_defined_vars in Hadd.
  rewrite (fold_on_defined_vars_defined _ _ level (Some menv) Hindexed)
    in Hadd.
  cbv beta iota in Hadd.
  exact (define_all_not_in_terms _ _ y menv menv' Hadd).
Qed.

(* [add_defined_vars], on a level that has its index. *)
Lemma add_defined_vars_defined (menv menv' : Meet_env.t)
    (level : Typing_env_level.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  level_indexed level ->
  add_defined_vars choices menv level = Some menv' ->
  defined_from (Typing_env_level.find_kind level)
    (Typing_env_level.variables_in_order level) menv menv'.
Proof.
  intros Hfits Hindexed Hadd. unfold add_defined_vars in Hadd.
  rewrite (fold_on_defined_vars_defined _ _ level (Some menv) Hindexed)
    in Hadd.
  cbv beta iota in Hadd.
  exact (define_all_defined _ _ menv menv' Hfits Hadd).
Qed.

End Define.

(* ======================================================================== *)
(* The environment of an alternative                                        *)
(* ======================================================================== *)

(* [stenv] is the environment [tenv] one scope further, after some of
   a meet there. *)
Section Scoped.

Variable tenv stenv : Typing_env.t.

Hypothesis Hfits : typing_env_fits tenv.
Hypothesis Hscoped_fits : typing_env_fits stenv.
Hypothesis Hscoped : tenv_extends (Typing_env.increment_scope tenv) stenv.

Let level := Typing_env.One_level.level (Typing_env.current_level stenv).

(* The level that is cut is the current one. *)
Lemma scoped_cut :
  Typing_env.cut stenv (Typing_env.current_scope tenv) = Some level.
Proof.
  destruct Hscoped as (Hscope & Hprev & _).
  unfold Typing_env.cut. rewrite Hscope.
  change (Typing_env.current_scope (Typing_env.increment_scope tenv))
    with (Scope.next (Typing_env.current_scope tenv)).
  unfold Scope.next.
  destruct (Z.leb_spec (Typing_env.current_scope tenv + 1)
              (Typing_env.current_scope tenv)) as [ Hle | _ ]; [ lia | ].
  rewrite Hprev.
  cbn [Typing_env.prev_levels Typing_env.increment_scope Typing_env.cut_loop].
  unfold Typing_env.current_scope. rewrite Z.ltb_irrefl. reflexivity.
Qed.

Lemma scoped_scope :
  (Typing_env.current_scope tenv
   < Typing_env.One_level.scope (Typing_env.current_level stenv))%Z.
Proof.
  destruct Hscoped as (Hscope & _).
  change (Typing_env.One_level.scope (Typing_env.current_level stenv))
    with (Typing_env.current_scope stenv).
  rewrite Hscope.
  change (Typing_env.current_scope (Typing_env.increment_scope tenv))
    with (Scope.next (Typing_env.current_scope tenv)).
  unfold Scope.next. lia.
Qed.

Lemma scoped_knows : knows_at_kinds tenv stenv.
Proof.
  split.
  - intros n Hmem. exact (tenv_extends_mem _ _ n Hscoped Hmem).
  - intros s kind Hs.
    apply (proj1 (proj2 (proj2 (proj2 (proj2 Hscoped))))).
    apply increment_scope_simple_of_kind. exact Hs.
Qed.

Lemma scoped_keeps_names : keeps_names tenv stenv.
Proof. exact (proj2 (proj2 (proj2 (proj2 (proj2 Hscoped))))). Qed.

Lemma scoped_scoped_env_of : scoped_env_of tenv stenv.
Proof.
  split; [ exact Hscoped_fits | ].
  split; [ exact (proj1 scoped_knows) | ].
  split; [ exact scoped_keeps_names | ].
  intros sym Hsym. unfold Typing_env.mem in *.
  rewrite (proj1 (proj2 (proj2 (proj2 Hscoped)))) in Hsym. exact Hsym.
Qed.

(* A variable that the level defines is of the kind that it defines it
   with. *)
Lemma scoped_defined_kind (x : variable) (k : flambda_kind) :
  lookup x (Typing_env_level.defined_vars level) = Some k ->
  simple_of_kind stenv (Var x) k.
Proof.
  intros Hdefined.
  destruct (cut_defined_variables_found stenv (Typing_env.current_scope tenv)
              level x k (proj1 (proj1 Hscoped_fits)) scoped_cut Hdefined)
    as (ty & binding_time_and_mode & Hlookup & Hkind).
  apply (simple_of_kind_name stenv (Name_var x)). left.
  exists ty, binding_time_and_mode. split; assumption.
Qed.

(* A variable that is new is defined by the level. *)
Lemma scoped_new_defined (x : variable) :
  Typing_env.mem None stenv (Name_var x) = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  is_bound x (Typing_env_level.defined_vars level) = true.
Proof.
  intros Hknown Hunknown.
  rewrite mem_var_is_bound in Hknown. unfold Keyed.is_bound in Hknown.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types stenv))
    as [ [ty binding_time_and_mode] | ] eqn:Hlookup;
    [ | discriminate Hknown ].
  destruct (proj1 (typing_env_well_formed_iff stenv)
              (proj1 (proj1 Hscoped_fits)))
    as (_ & (_ & _ & _ & _ & _ & _ & Hdefined) & _).
  destruct (Hdefined x ty binding_time_and_mode Hlookup)
    as (defining & Hin & Hbound).
  unfold all_levels in Hin. cbn [map] in Hin.
  destruct Hin as [ <- | Hin ]; [ exact Hbound | ].
  exfalso. rewrite (proj1 (proj2 Hscoped)) in Hin.
  change (defining ∈ all_levels tenv) in Hin.
  destruct (proj1 (typing_env_well_formed_iff tenv) (proj1 (proj1 Hfits)))
    as (_ & (_ & Hbound_tenv & _) & _).
  unfold is_bound in Hbound.
  destruct (lookup x (Typing_env_level.defined_vars defining)) as [ k | ]
    eqn:Hk; [ | discriminate Hbound ].
  destruct (Hbound_tenv defining x k Hin Hk)
    as (ty' & binding_time_and_mode' & Hlookup' & _).
  rewrite mem_var_is_bound in Hunknown. unfold Keyed.is_bound in Hunknown.
  rewrite Hlookup' in Hunknown. discriminate Hunknown.
Qed.

Lemma scoped_level_indexed : level_indexed level.
Proof.
  exact (proj1 (cut_indexed stenv (Typing_env.current_scope tenv) level
                  (proj1 (proj1 Hscoped_fits)) scoped_cut)).
Qed.

(* An equation of the level is the type that the environment has. *)
Lemma scoped_equation (n : name) (ty : flambda_type) :
  Keyed.lookup name_eqb n (Typing_env_level.equations level) = Some ty ->
  ∃ binding_time_and_mode : Binding_time.With_name_mode.t,
    Keyed.lookup name_eqb n (Typing_env.names_to_types stenv)
    = Some (ty, binding_time_and_mode).
Proof.
  exact (cut_equations_found stenv (Typing_env.current_scope tenv) level n ty
           (proj2 (proj1 Hscoped_fits)) scoped_cut).
Qed.

Lemma scoped_equation_kind (n : name) (ty : flambda_type) :
  Keyed.lookup name_eqb n (Typing_env_level.equations level) = Some ty ->
  simple_of_kind stenv (simple_of_name n) (kind ty).
Proof.
  intros Hlookup.
  destruct (scoped_equation n ty Hlookup) as [binding_time_and_mode Hfound].
  apply simple_of_kind_name. left. exists ty, binding_time_and_mode.
  split; [ exact Hfound | reflexivity ].
Qed.

(* The extension made of the level fits the environment. *)
Lemma scoped_extension_fits : extension_fits stenv (extension_of_level level).
Proof.
  unfold extension_of_level. split; [ | split ].
  - apply env_extension_well_formed_iff. split.
    + exact (keyed_compact_distinct name name_eqb name_eqb_eq _ _).
    + intros n ty Hlookup.
      rewrite (keyed_lookup_compact name name_eqb name_eqb_eq) in Hlookup.
      destruct (scoped_equation n ty Hlookup)
        as [binding_time_and_mode Hfound].
      exact (proj1 (fits_types stenv n ty binding_time_and_mode Hscoped_fits
                      Hfound)).
  - intros n ty Hlookup.
    rewrite (keyed_lookup_compact name name_eqb name_eqb_eq) in Hlookup.
    split.
    + exact (simple_of_kind_mem stenv n (kind ty) (proj1 Hscoped_fits)
               (scoped_equation_kind n ty Hlookup)).
    + destruct (scoped_equation n ty Hlookup)
        as [binding_time_and_mode Hfound].
      exact (fits_types stenv n ty binding_time_and_mode Hscoped_fits Hfound).
  - intros n ty Hlookup.
    rewrite (keyed_lookup_compact name name_eqb name_eqb_eq) in Hlookup.
    exact (scoped_equation_kind n ty Hlookup).
Qed.

(* The extension made of the level is true, where the names of its
   equations have values. *)
Lemma scoped_extension_satisfied (statics : static_env) (valuation : env) :
  typing_env_satisfied_strictly statics valuation stenv ->
  equations_valued stenv (Typing_env.current_scope tenv) valuation ->
  env_extension_satisfied statics valuation (extension_of_level level).
Proof.
  intros (_ & Htypes & _) Hvalued. unfold extension_of_level.
  apply (env_extension_lookup statics valuation). intros n ty Hlookup.
  rewrite (keyed_lookup_compact name name_eqb name_eqb_eq) in Hlookup.
  destruct (scoped_equation n ty Hlookup) as [binding_time_and_mode Hfound].
  destruct (Hvalued (Typing_env.current_level stenv) n ty
              (or_introl eq_refl) scoped_scope
              (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hlookup))
    as [v Hv].
  exists v. split; [ exact Hv | ].
  destruct (Htypes n ty binding_time_and_mode v Hfound Hv)
    as [ [[] _] | Hty ]; exact Hty.
Qed.

(* A variable that the level defines has a type, of the kind. *)
Lemma scoped_defined_type (x : variable) (k : flambda_kind) :
  lookup x (Typing_env_level.defined_vars level) = Some k ->
  ∃ (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types stenv)
    = Some (ty, binding_time_and_mode)
    ∧ kind ty = k.
Proof.
  exact (cut_defined_variables_found stenv (Typing_env.current_scope tenv)
           level x k (proj1 (proj1 Hscoped_fits)) scoped_cut).
Qed.

(* A variable that the fold on the level comes to, with a kind, is
   known. *)
Lemma scoped_in_order_known (x : variable) (k : flambda_kind) :
  Typing_env_level.find_kind level x = Some k ->
  Typing_env.mem None stenv (Name_var x) = true.
Proof.
  intros Hk.
  exact (simple_of_kind_mem stenv (Name_var x) k (proj1 Hscoped_fits)
           (scoped_defined_kind x k Hk)).
Qed.

(* A variable that is new is one that the fold on the level comes to,
   and it has a kind there. *)
Lemma scoped_new_in_order (x : variable) :
  Typing_env.mem None stenv (Name_var x) = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  x ∈ Typing_env_level.variables_in_order level
  ∧ ∃ k : flambda_kind, Typing_env_level.find_kind level x = Some k.
Proof.
  intros Hknown Hunknown.
  assert (Hbound := scoped_new_defined x Hknown Hunknown). split.
  - apply (variables_in_order_defined level x scoped_level_indexed).
    exact Hbound.
  - unfold is_bound in Hbound. unfold Typing_env_level.find_kind.
    destruct (lookup x (Typing_env_level.defined_vars level)) as [ k | ];
      [ exists k; reflexivity | discriminate Hbound ].
Qed.

Lemma scoped_symbol_known (sym : symbol) :
  Typing_env.mem None stenv (Name_symbol sym)
  = Typing_env.mem None tenv (Name_symbol sym).
Proof.
  unfold Typing_env.mem. rewrite (proj1 (proj2 (proj2 (proj2 Hscoped)))).
  reflexivity.
Qed.

(* What an environment knows after the variables of the level have
   been defined in it, if it knew what [tenv] knows and had what it
   knew of [stenv] at the kinds of [stenv]: all that [stenv] knows, at
   its kinds. *)
Lemma scoped_knows_after (menv0 menv1 : Meet_env.t) :
  knows_at_kinds tenv (Meet_env.typing_env menv0) ->
  (∀ (s : simple) (kind : flambda_kind),
     Typing_env.mem_simple None (Meet_env.typing_env menv0) s = true ->
     simple_of_kind stenv s kind ->
     simple_of_kind (Meet_env.typing_env menv0) s kind) ->
  defined_from (Typing_env_level.find_kind level)
    (Typing_env_level.variables_in_order level) menv0 menv1 ->
  knows_at_kinds stenv (Meet_env.typing_env menv1).
Proof.
  intros [Hmem0 Hkind0] Hagree (Hextends & _ & _ & _ & Hkind & Hknown & _).
  assert (Htenv := meet_env_extends_tenv _ _ Hextends).
  assert (Hnew :
    ∀ x : variable,
      Typing_env.mem None stenv (Name_var x) = true ->
      Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x) = false ->
      x ∈ Typing_env_level.variables_in_order level
      ∧ ∃ k : flambda_kind, Typing_env_level.find_kind level x = Some k).
  { intros x Hx Hunknown. apply scoped_new_in_order; [ exact Hx | ].
    destruct (Typing_env.mem None tenv (Name_var x)) eqn:Hold;
      [ | reflexivity ].
    rewrite (Hmem0 _ Hold) in Hunknown. discriminate Hunknown. }
  split.
  - intros n Hn. destruct n as [ x | sym ].
    + destruct (Typing_env.mem None (Meet_env.typing_env menv0) (Name_var x))
        eqn:Hx.
      * exact (tenv_extends_mem _ _ _ Htenv Hx).
      * destruct (Hnew x Hn Hx) as [Hin [k Hk]]. exact (Hknown x k Hin Hk).
    + apply (tenv_extends_mem _ _ _ Htenv). apply Hmem0.
      rewrite <- scoped_symbol_known. exact Hn.
  - intros s kind Hs.
    destruct (Typing_env.mem_simple None (Meet_env.typing_env menv0) s)
      eqn:Hknown0.
    + exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv)))) s kind
               (Hagree s kind Hknown0 Hs)).
    + destruct s as [ x | sym | c ]; [ | | discriminate Hknown0 ].
      * destruct (Hnew x
                    (simple_of_kind_mem stenv (Name_var x) kind
                       (proj1 Hscoped_fits) Hs)
                    Hknown0) as [Hin [k Hk]].
        assert (Hsame : k = kind).
        { exact (simple_of_kind_unique stenv (Var x) k kind
                   (scoped_defined_kind x k Hk) Hs). }
        subst k. exact (Hkind x kind Hin Hk Hknown0).
      * exfalso.
        assert (Hsym := simple_of_kind_mem stenv (Name_symbol sym) kind
                          (proj1 Hscoped_fits) Hs).
        rewrite scoped_symbol_known in Hsym.
        cbn [Typing_env.mem_simple] in Hknown0.
        rewrite (Hmem0 _ Hsym) in Hknown0. discriminate Hknown0.
Qed.

(* A value that a valuation that satisfies [stenv] has for a variable
   that the level defines is of the kind. *)
Lemma scoped_defined_value_kind (statics : static_env) (valuation : env)
    (x : variable) (k : flambda_kind) (v : value) :
  typing_env_satisfied_strictly statics valuation stenv ->
  Typing_env_level.find_kind level x = Some k ->
  lookup x valuation = Some v ->
  kind_of_value v = k.
Proof.
  intros (_ & Htypes & _) Hk Hv.
  destruct (scoped_defined_type x k Hk)
    as (ty & binding_time_and_mode & Hlookup & <-).
  destruct (Htypes (Name_var x) ty binding_time_and_mode v Hlookup Hv)
    as [ [[] _] | Hty ].
  exact (type_satisfied_kind statics valuation v ty Hty).
Qed.

End Scoped.

(* ======================================================================== *)
(* [meet_disjunction], in parts                                             *)
(* ======================================================================== *)

(* The functions that [meet_disjunction] defines for itself. *)
Definition add_optional_extension (mh : meet_expanded_head)
    (env : Or_bottom.t Meet_env.t) (extension : option env_extension)
    : option (Or_bottom.t Meet_env.t) :=
  match extension, env with
  | None, _ => Some env
  | Some _, Or_bottom.Bottom => Some Or_bottom.Bottom
  | Some extension, Or_bottom.Ok env =>
    Meet_env.add_env_extension_strict env extension mh
  end.

Definition meet_side {C : Type}
    (meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (env : Or_bottom.t Meet_env.t) (val1 val2 : C)
    : option (Meet_result.t C) :=
  match env with
  | Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
  | Or_bottom.Ok env =>
    let* result := meet_c env val1 val2 in
    match result with
    | Meet_result.Bottom r => Some (Meet_result.Bottom r)
    | Meet_result.Ok result env =>
      match Meet_env.final_typing_env_strict env with
      | Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
      | Or_bottom.Ok _ => Some (Meet_result.Ok result env)
      end
    end
  end.

(* What one side added goes to the environment of the meet. *)
Definition direct_return {R : Type} (choices : meet_choices)
    (mh : meet_expanded_head) (initial_env : Meet_env.t) (join_scope : Scope.t)
    (result : meet_return_value R) (scoped_env : Meet_env.t)
    : option (Meet_result.t R) :=
  let* level := Typing_env.cut (Meet_env.typing_env scoped_env) join_scope in
  let* initial_env :=
    add_defined_vars choices
      (Meet_env.with_names_to_avoid initial_env
         (Meet_env.names_to_avoid scoped_env))
      level
  in
  let ext := extension_of_level level in
  let* added := Meet_env.add_env_extension_strict initial_env ext mh in
  match added with
  | Or_bottom.Bottom =>
    Some (Meet_result.Bottom (map_return_value (fun _ => tt) result))
  | Or_bottom.Ok env => Some (Meet_result.Ok result env)
  end.

(* The two sides are joined. *)
Definition join_return {A B : Type} (join : join_scoped_envs)
    (choices : meet_choices) (mh : meet_expanded_head)
    (initial_env : Meet_env.t) (join_scope : Scope.t)
    (val_a1 : A) (val_b1 : B) (extensions1 : variant_extensions)
    (val_a2 : A) (val_b2 : B) (extensions2 : variant_extensions)
    (a_result : meet_return_value A) (env_a : Meet_env.t)
    (b_result : meet_return_value B) (env_b : Meet_env.t)
    : option (Meet_result.t (A * B * variant_extensions)) :=
  let initial_tenv := Meet_env.typing_env initial_env in
  let tenv_a := Meet_env.final_typing_env env_a in
  let tenv_b := Meet_env.final_typing_env env_b in
  let* result_env :=
    join mh join_scope
      (Meet_env.with_names_to_avoid initial_env (Meet_env.names_to_avoid env_b))
      initial_tenv [tenv_a; tenv_b]
  in
  let* when_a_level := Typing_env.cut tenv_a join_scope in
  let* when_b_level := Typing_env.cut tenv_b join_scope in
  let* result_env := add_defined_vars choices result_env when_a_level in
  let* result_env := add_defined_vars choices result_env when_b_level in
  let extensions :=
    if Typing_env_level.is_empty when_a_level
       && Typing_env_level.is_empty when_b_level
    then No_extensions
    else
      Ext (extension_of_level when_a_level) (extension_of_level when_b_level)
  in
  let env_extension_result : meet_return_value unit :=
    match extensions, extensions1, extensions2 with
    | No_extensions, No_extensions, No_extensions => Both_inputs
    | No_extensions, No_extensions, Ext _ _ => Left_input
    | No_extensions, Ext _ _, No_extensions => Right_input
    | (No_extensions | Ext _ _), _, _ => New_result tt
    end
  in
  let result :=
    combine_meet_return_values
      (combine_meet_return_values a_result b_result (fun _ => tt))
      env_extension_result
      (fun _ =>
         (extract_value a_result val_a1 val_a2,
          extract_value b_result val_b1 val_b2, extensions))
  in
  Some (Meet_result.Ok result result_env).

Lemma meet_disjunction_eq {A B : Type} (join : join_scoped_envs)
    (choices : meet_choices) (mh : meet_expanded_head)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) (initial_env : Meet_env.t)
    (a1 : A) (b1 : B) (e1 : variant_extensions) (a2 : A) (b2 : B)
    (e2 : variant_extensions) :
  meet_disjunction join choices mh meet_a meet_b bottom_a bottom_b initial_env
    a1 b1 e1 a2 b2 e2
  = let join_scope :=
      Typing_env.current_scope (Meet_env.typing_env initial_env)
    in
    let env := snd (Meet_env.enter_scope initial_env) in
    let* env_a :=
      add_optional_extension mh (Or_bottom.Ok env) (extension_when_a e1)
    in
    let* env_a := add_optional_extension mh env_a (extension_when_a e2) in
    let* a_result := meet_side meet_a env_a a1 a2 in
    let env :=
      match a_result with
      | Meet_result.Bottom _ => env
      | Meet_result.Ok _ env_a =>
        Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env_a)
      end
    in
    let* env_b :=
      add_optional_extension mh (Or_bottom.Ok env) (extension_when_b e1)
    in
    let* env_b := add_optional_extension mh env_b (extension_when_b e2) in
    let* b_result := meet_side meet_b env_b b1 b2 in
    match a_result, b_result with
    | Meet_result.Bottom r1, Meet_result.Bottom r2 =>
      Some
        (Meet_result.Bottom (combine_meet_return_values r1 r2 (fun _ => tt)))
    | Meet_result.Ok a_result env_a, Meet_result.Bottom b =>
      direct_return choices mh initial_env join_scope
        (combine_meet_return_values a_result b
           (fun _ =>
              (extract_value a_result a1 a2, bottom_b tt, No_extensions)))
        env_a
    | Meet_result.Bottom a, Meet_result.Ok b_result env_b =>
      direct_return choices mh initial_env join_scope
        (combine_meet_return_values a b_result
           (fun _ =>
              (bottom_a tt, extract_value b_result b1 b2, No_extensions)))
        env_b
    | Meet_result.Ok a_result env_a, Meet_result.Ok b_result env_b =>
      join_return join choices mh initial_env join_scope a1 b1 e1 a2 b2 e2
        a_result env_a b_result env_b
    end.
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* One side                                                                 *)
(* ======================================================================== *)

Section Side.

Context {C : Type}.

Variable fits_c : Typing_env.t -> C -> Prop.
Variable satisfied_c : static_env -> env -> C -> Prop.
Variable mh : meet_expanded_head.
Variable meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C).

Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.
Hypothesis Hstable : part_stable fits_c satisfied_c.
Hypothesis Hmeet_c : part_meet_ok fits_c satisfied_c meet_c.

Lemma add_optional_extension_keeps (menv menv' : Meet_env.t)
    (extension : option env_extension) :
  typing_env_fits (Meet_env.typing_env menv) ->
  optional_extension_fits (Meet_env.typing_env menv) extension ->
  add_optional_extension mh (Or_bottom.Ok menv) extension
  = Some (Or_bottom.Ok menv') ->
  meet_env_kept_by_meet menv menv'.
Proof.
  intros Hfits Hextension Hadd.
  destruct extension as [ extension | ]; cbn in Hadd.
  - destruct Hextension as (_ & Hextension & Hkinded).
    exact (add_env_extension_strict_keeps_for mh menv menv' extension Hkeeps
             Hfits Hextension Hkinded Hadd).
  - injection Hadd as <-.
    exact (meet_env_steps_kept _ _ (conj (meet_env_steps_refl menv) Hfits)).
Qed.

Lemma add_optional_extension_sound (statics : static_env) (valuation : env)
    (menv : Meet_env.t) (extension : option env_extension)
    (result : Or_bottom.t Meet_env.t) :
  typing_env_fits (Meet_env.typing_env menv) ->
  optional_extension_fits (Meet_env.typing_env menv) extension ->
  statics_valid statics ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv) ->
  optional_extension_satisfied statics valuation extension ->
  add_optional_extension mh (Or_bottom.Ok menv) extension = Some result ->
  meet_env_sound statics valuation menv result.
Proof.
  intros Hfits Hextension Hstatics Hsatisfied Htrue Hadd.
  destruct extension as [ extension | ]; cbn in Hadd.
  - destruct Hextension as (_ & Hextension & Hkinded).
    exact (meet_env_add_env_extension_strict_sound_for mh statics valuation menv
             extension result Hkeeps Hsound Hfits Hextension Hkinded
             Hstatics Hsatisfied Htrue Hadd).
  - injection Hadd as <-. exists menv. split; [ reflexivity | ].
    exact (meet_env_added_refl statics valuation menv Hsatisfied).
Qed.

Variable base : Meet_env.t.
Variable extension1 extension2 : option env_extension.
Variable c1 c2 : C.

Hypothesis Hfits : typing_env_fits (Meet_env.typing_env base).
Hypothesis Hextension1 :
  optional_extension_fits (Meet_env.typing_env base) extension1.
Hypothesis Hextension2 :
  optional_extension_fits (Meet_env.typing_env base) extension2.
Hypothesis Hc1 : fits_c (Meet_env.typing_env base) c1.
Hypothesis Hc2 : fits_c (Meet_env.typing_env base) c2.

(* What one side keeps. *)
Lemma side_keeps (env1 env2 : Or_bottom.t Meet_env.t)
    (r : meet_return_value C) (env_c : Meet_env.t) :
  add_optional_extension mh (Or_bottom.Ok base) extension1 = Some env1 ->
  add_optional_extension mh env1 extension2 = Some env2 ->
  meet_side meet_c env2 c1 c2 = Some (Meet_result.Ok r env_c) ->
  meet_env_extends base env_c
  ∧ typing_env_fits (Meet_env.typing_env env_c)
  ∧ fits_c (Meet_env.typing_env env_c) (extract_value r c1 c2)
  ∧ new_names_avoided base env_c.
Proof.
  intros Hadd1 Hadd2 Hside.
  destruct env2 as [ menv2 | ]; [ | discriminate Hside ].
  destruct env1 as [ menv1 | ].
  2:{ destruct extension2; discriminate Hadd2. }
  destruct (add_optional_extension_keeps base menv1 extension1 Hfits
              Hextension1 Hadd1) as [[Hextends1 Hfits1] Hnames1].
  destruct (add_optional_extension_keeps menv1 menv2 extension2 Hfits1
              (optional_extension_fits_kept _ _ _
                 (meet_env_extends_knows _ _ Hextends1) Hextension2) Hadd2)
    as [[Hextends2 Hfits2] Hnames2].
  assert (Hextends12 := meet_env_extends_trans _ _ _ Hextends1 Hextends2).
  unfold meet_side in Hside.
  destruct (meet_c menv2 c1 c2) as [ [ r0 | r0 menv3 ] | ] eqn:Hmeet;
    try discriminate Hside.
  cbn in Hside. injection Hside as <- <-.
  destruct (Hmeet_c menv2 c1 c2 _ Hfits2
              (proj1 Hstable base menv2 c1 Hextends12 Hc1)
              (proj1 Hstable base menv2 c2 Hextends12 Hc2) Hmeet)
    as [Hkept _].
  destruct (Hkept r0 menv3 eq_refl)
    as (Hextends3 & Hfits3 & Hanswer & Hnames3).
  split; [ exact (meet_env_extends_trans _ _ _ Hextends12 Hextends3) | ].
  split; [ exact Hfits3 | ]. split; [ exact Hanswer | ].
  exact (new_names_avoided_trans _ _ _ Hextends12 Hextends3
           (new_names_avoided_trans _ _ _ Hextends1 Hextends2 Hnames1 Hnames2)
           Hnames3).
Qed.

(* One side loses no value. *)
Lemma side_sound (statics : static_env) (valuation : env)
    (env1 env2 : Or_bottom.t Meet_env.t) (result : Meet_result.t C) :
  statics_valid statics ->
  typing_env_satisfied_strictly statics valuation (Meet_env.typing_env base) ->
  optional_extension_satisfied statics valuation extension1 ->
  optional_extension_satisfied statics valuation extension2 ->
  satisfied_c statics valuation c1 ->
  satisfied_c statics valuation c2 ->
  add_optional_extension mh (Or_bottom.Ok base) extension1 = Some env1 ->
  add_optional_extension mh env1 extension2 = Some env2 ->
  meet_side meet_c env2 c1 c2 = Some result ->
  ∃ (r : meet_return_value C) (env_c : Meet_env.t) (valuation' : env),
    result = Meet_result.Ok r env_c
    ∧ extended_from (Meet_env.typing_env base) valuation valuation'
    ∧ typing_env_satisfied_strictly statics valuation'
        (Meet_env.typing_env env_c)
    ∧ satisfied_c statics valuation' (extract_value r c1 c2)
    ∧ keeps_equations_valued (Meet_env.typing_env base)
        (Meet_env.typing_env env_c) valuation valuation'.
Proof.
  intros Hstatics Hsatisfied Htrue1 Htrue2 Hv1 Hv2 Hadd1 Hadd2 Hside.
  destruct (add_optional_extension_sound statics valuation base extension1
              env1 Hfits Hextension1 Hstatics Hsatisfied Htrue1 Hadd1)
    as (menv1 & -> & valuation1 & Hextended1 & Hsatisfied1 & Hvalued1).
  destruct (add_optional_extension_keeps base menv1 extension1 Hfits
              Hextension1 Hadd1) as [[Hextends1 Hfits1] _].
  assert (Hknows1 := meet_env_extends_knows _ _ Hextends1).
  assert (Hextension2' := optional_extension_fits_kept _ _ _ Hknows1
                            Hextension2).
  destruct (add_optional_extension_sound statics valuation1 menv1 extension2
              env2 Hfits1 Hextension2' Hstatics Hsatisfied1
              (optional_extension_satisfied_extended statics _ valuation
                 valuation1 extension2 Hextension2 Hextended1 Htrue2) Hadd2)
    as (menv2 & -> & valuation2 & Hextended2 & Hsatisfied2 & Hvalued2).
  destruct (add_optional_extension_keeps menv1 menv2 extension2 Hfits1
              Hextension2' Hadd2) as [[Hextends2 Hfits2] _].
  assert (Hextends12 := meet_env_extends_trans _ _ _ Hextends1 Hextends2).
  assert (Hextended12 :
    extended_from (Meet_env.typing_env base) valuation valuation2).
  { exact (extended_from_trans _ _ _ _ _
             (proj2 (proj2 (proj2 (proj2 (proj2
                (meet_env_extends_tenv _ _ Hextends1))))))
             Hextended1 Hextended2). }
  unfold meet_side in Hside.
  destruct (meet_c menv2 c1 c2) as [ result0 | ] eqn:Hmeet;
    [ | discriminate Hside ].
  destruct (Hmeet_c menv2 c1 c2 result0 Hfits2
              (proj1 Hstable base menv2 c1 Hextends12 Hc1)
              (proj1 Hstable base menv2 c2 Hextends12 Hc2) Hmeet)
    as [_ Hlost].
  destruct (Hlost statics valuation2 Hstatics Hsatisfied2
              (proj2 Hstable statics _ valuation valuation2 c1 Hc1 Hextended12
                 Hv1)
              (proj2 Hstable statics _ valuation valuation2 c2 Hc2 Hextended12
                 Hv2))
    as (r & menv3 & valuation3 & -> & Hextended3 & Hsatisfied3 & Hanswer
        & Hvalued3).
  cbn in Hside. injection Hside as <-.
  exists r, menv3, valuation3. split; [ reflexivity | ]. split.
  { exact (extended_from_trans _ _ _ _ _
             (proj2 (proj2 (proj2 (proj2 (proj2
                (meet_env_extends_tenv _ _ Hextends12))))))
             Hextended12 Hextended3). }
  split; [ exact Hsatisfied3 | ]. split; [ exact Hanswer | ].
  exact (keeps_equations_valued_trans _ _ _ _ _ _
           (keeps_equations_valued_trans _ _ _ _ _ _ Hvalued1 Hvalued2)
           Hvalued3).
Qed.

End Side.

(* ======================================================================== *)
(* What one side added goes to the environment of the meet                  *)
(* ======================================================================== *)

Lemma meet_env_extends_avoid (menv menv' : Meet_env.t) :
  meet_env_extends menv menv' ->
  ∀ x : variable,
    among x (Meet_env.names_to_avoid menv) = true ->
    among x (Meet_env.names_to_avoid menv') = true.
Proof. intros (_ & _ & _ & _ & _ & _ & Havoid & _). exact Havoid. Qed.

(* [scoped] is the environment of an alternative of a meet in
   [initial_env]: an environment [base], which is [initial_env] one
   scope further, with names to avoid that include its own, after some
   of a meet. *)
Section Alternative.

Variable choices : meet_choices.
Hypothesis Hchoice : define_again choices = false.
Variable mh : meet_expanded_head.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.

Variable initial_env base scoped : Meet_env.t.

Local Abbreviation tenv := (Meet_env.typing_env initial_env).
Local Abbreviation stenv := (Meet_env.typing_env scoped).
Local Abbreviation level :=
  (Typing_env.One_level.level (Typing_env.current_level stenv)).

Hypothesis Hfits : typing_env_fits tenv.
Hypothesis Hbase :
  Meet_env.typing_env base = Typing_env.increment_scope tenv.
Hypothesis Hbase_avoid :
  ∀ x : variable,
    among x (Meet_env.names_to_avoid initial_env) = true ->
    among x (Meet_env.names_to_avoid base) = true.
Hypothesis Hextends : meet_env_extends base scoped.
Hypothesis Hscoped_fits : typing_env_fits stenv.
Hypothesis Hnames : new_names_avoided base scoped.

Lemma alternative_scoped :
  tenv_extends (Typing_env.increment_scope tenv) stenv.
Proof. rewrite <- Hbase. exact (meet_env_extends_tenv _ _ Hextends). Qed.

(* A variable of the alternative that is new is to be avoided from
   then on. *)
Lemma alternative_new_avoided (x : variable) :
  Typing_env.mem None stenv (Name_var x) = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  among x (Meet_env.names_to_avoid base) = false
  ∧ among x (Meet_env.names_to_avoid scoped) = true
  ∧ Typing_env.mem (Some Name_mode.normal) stenv (Name_var x) = false.
Proof.
  intros Hknown Hunknown. apply (Hnames x Hknown). rewrite Hbase.
  exact Hunknown.
Qed.

Lemma alternative_avoid_initial (x : variable) :
  among x (Meet_env.names_to_avoid initial_env) = true ->
  among x (Meet_env.names_to_avoid scoped) = true.
Proof.
  intros Hx. exact (meet_env_extends_avoid _ _ Hextends x (Hbase_avoid x Hx)).
Qed.

Lemma initial_extends_avoiding :
  meet_env_extends initial_env
    (Meet_env.with_names_to_avoid initial_env
       (Meet_env.names_to_avoid scoped)).
Proof.
  repeat split; try reflexivity.
  - intros s kind Hs. exact Hs.
  - exact alternative_avoid_initial.
  - apply keeps_names_refl.
Qed.

(* [direct_return]: what it keeps. *)
Lemma direct_return_keeps {R : Type} (r r' : meet_return_value R)
    (result_env : Meet_env.t) :
  direct_return choices mh initial_env (Typing_env.current_scope tenv) r scoped
  = Some (Meet_result.Ok r' result_env) ->
  r' = r
  ∧ meet_env_extends initial_env result_env
  ∧ typing_env_fits (Meet_env.typing_env result_env)
  ∧ new_names_avoided initial_env result_env
  ∧ knows_at_kinds stenv (Meet_env.typing_env result_env).
Proof.
  intros Hreturn. unfold direct_return in Hreturn.
  rewrite (scoped_cut tenv stenv alternative_scoped) in Hreturn.
  cbv beta iota in Hreturn.
  destruct (add_defined_vars choices
              (Meet_env.with_names_to_avoid initial_env
                 (Meet_env.names_to_avoid scoped))
              level) as [ menv1 | ] eqn:Hdefined; [ | discriminate Hreturn ].
  cbv beta iota zeta in Hreturn.
  destruct (Meet_env.add_env_extension_strict menv1 (extension_of_level level)
              mh) as [ [ menv2 | ] | ] eqn:Hadd; try discriminate Hreturn.
  injection Hreturn as <- <-.
  set (menv0 :=
    Meet_env.with_names_to_avoid initial_env (Meet_env.names_to_avoid scoped))
    in *.
  assert (Hdef :=
    add_defined_vars_defined choices Hchoice menv0 menv1 level Hfits
      (scoped_level_indexed tenv stenv Hscoped_fits alternative_scoped)
      Hdefined).
  assert (Hknows1 : knows_at_kinds stenv (Meet_env.typing_env menv1)).
  { apply (scoped_knows_after tenv stenv Hfits Hscoped_fits alternative_scoped
             menv0 menv1); [ | | exact Hdef ].
    - split; [ intros n Hn; exact Hn | intros s kind Hs; exact Hs ].
    - intros s kind Hknown Hs.
      exact (scoped_env_of_kind tenv stenv s kind
               (scoped_scoped_env_of tenv stenv Hscoped_fits
                  alternative_scoped)
               Hknown Hs). }
  destruct Hdef as (Hextends01 & Hfits1 & Havoid01 & Hmem01 & _).
  destruct (extension_fits_kept _ _ _ Hknows1
              (scoped_extension_fits tenv stenv Hscoped_fits
                 alternative_scoped))
    as (_ & Hextension & Hkinded).
  destruct (add_env_extension_strict_keeps_for mh menv1 menv2 _ Hkeeps Hfits1
              Hextension Hkinded Hadd) as [[Hextends12 Hfits2] Hnames12].
  split; [ reflexivity | ]. split.
  { exact (meet_env_extends_trans _ _ _ initial_extends_avoiding
             (meet_env_extends_trans _ _ _ Hextends01 Hextends12)). }
  split; [ exact Hfits2 | ]. split.
  - intros x Hknown Hunknown.
    destruct (Typing_env.mem None (Meet_env.typing_env menv1) (Name_var x))
      eqn:Hx.
    + destruct (Hmem01 x Hx) as [ Hold | [Hin Hsome] ].
      { change (Typing_env.mem None tenv (Name_var x) = true) in Hold.
        rewrite Hold in Hunknown. discriminate Hunknown. }
      destruct (Typing_env_level.find_kind level x) as [ k | ] eqn:Hk;
        [ | exfalso; apply Hsome; reflexivity ].
      destruct (alternative_new_avoided x
                  (scoped_in_order_known tenv stenv Hscoped_fits
                     alternative_scoped x k Hk)
                  Hunknown) as (Hnot & Hamong & _).
      split; [ | split ].
      * destruct (among x (Meet_env.names_to_avoid initial_env)) eqn:Ha;
          [ | reflexivity ].
        rewrite (Hbase_avoid x Ha) in Hnot. discriminate Hnot.
      * apply (meet_env_extends_avoid _ _ Hextends12). rewrite Havoid01.
        exact Hamong.
      * rewrite (meet_env_extends_mode _ _ _ x Hextends12 Hx).
        apply (add_defined_vars_not_in_terms choices menv0 menv1 level x
                 (scoped_level_indexed tenv stenv Hscoped_fits
                    alternative_scoped)
                 Hdefined).
        exact (unknown_not_in_terms tenv x Hunknown).
    + destruct (Hnames12 x Hknown Hx) as (Hnot & Hamong & Hterms).
      split; [ | split; [ exact Hamong | exact Hterms ] ].
      destruct (among x (Meet_env.names_to_avoid initial_env)) eqn:Ha;
        [ | reflexivity ].
      rewrite Havoid01 in Hnot.
      change (among x (Meet_env.names_to_avoid scoped) = false) in Hnot.
      rewrite (alternative_avoid_initial x Ha) in Hnot. discriminate Hnot.
  - exact (knows_at_kinds_trans _ _ _ Hknows1
             (meet_env_extends_knows _ _ Hextends12)).
Qed.

(* [direct_return]: no value is lost. *)
Lemma direct_return_sound {R : Type} (statics : static_env)
    (valuation valuation_s : env) (r : meet_return_value R)
    (result : Meet_result.t R) :
  statics_valid statics ->
  typing_env_satisfied_strictly statics valuation tenv ->
  extended_from tenv valuation valuation_s ->
  typing_env_satisfied_strictly statics valuation_s stenv ->
  keeps_equations_valued (Meet_env.typing_env base) stenv valuation
    valuation_s ->
  direct_return choices mh initial_env (Typing_env.current_scope tenv) r scoped
  = Some result ->
  ∃ (result_env : Meet_env.t) (valuation' : env),
    result = Meet_result.Ok r result_env
    ∧ extended_from stenv valuation_s valuation'
    ∧ typing_env_satisfied_strictly statics valuation'
        (Meet_env.typing_env result_env)
    ∧ keeps_equations_valued tenv (Meet_env.typing_env result_env) valuation
        valuation'.
Proof.
  intros Hstatics Hsatisfied Hextended Hsatisfied_s Hvalued_s Hreturn.
  unfold direct_return in Hreturn.
  rewrite (scoped_cut tenv stenv alternative_scoped) in Hreturn.
  cbv beta iota in Hreturn.
  destruct (add_defined_vars choices
              (Meet_env.with_names_to_avoid initial_env
                 (Meet_env.names_to_avoid scoped))
              level) as [ menv1 | ] eqn:Hdefined; [ | discriminate Hreturn ].
  cbv beta iota zeta in Hreturn.
  destruct (Meet_env.add_env_extension_strict menv1 (extension_of_level level)
              mh) as [ added | ] eqn:Hadd; [ | discriminate Hreturn ].
  set (menv0 :=
    Meet_env.with_names_to_avoid initial_env (Meet_env.names_to_avoid scoped))
    in *.
  assert (Hdef :=
    add_defined_vars_defined choices Hchoice menv0 menv1 level Hfits
      (scoped_level_indexed tenv stenv Hscoped_fits alternative_scoped)
      Hdefined).
  assert (Hknows1 : knows_at_kinds stenv (Meet_env.typing_env menv1)).
  { apply (scoped_knows_after tenv stenv Hfits Hscoped_fits alternative_scoped
             menv0 menv1); [ | | exact Hdef ].
    - split; [ intros n Hn; exact Hn | intros s kind Hs; exact Hs ].
    - intros s kind Hknown Hs.
      exact (scoped_env_of_kind tenv stenv s kind
               (scoped_scoped_env_of tenv stenv Hscoped_fits
                  alternative_scoped)
               Hknown Hs). }
  destruct Hdef as (Hextends01 & Hfits1 & _ & _ & _ & _ & Hsatisfied01
                    & Hvalued01).
  assert (Hsatisfied0 : typing_env_satisfied_strictly statics valuation_s tenv).
  { exact (satisfied_strictly_extended statics tenv valuation valuation_s
             (proj1 (proj1 Hfits)) Hextended Hsatisfied). }
  assert (Hsatisfied1 :
    typing_env_satisfied_strictly statics valuation_s
      (Meet_env.typing_env menv1)).
  { apply (Hsatisfied01 statics valuation_s Hsatisfied0).
    intros y k v _ Hk _ Hv.
    exact (scoped_defined_value_kind tenv stenv Hscoped_fits
             alternative_scoped statics valuation_s y k v Hsatisfied_s Hk
             Hv). }
  assert (Htrue :
    env_extension_satisfied statics valuation_s (extension_of_level level)).
  { apply (scoped_extension_satisfied tenv stenv Hscoped_fits
             alternative_scoped statics valuation_s Hsatisfied_s).
    apply Hvalued_s. rewrite Hbase.
    exact (increment_scope_valued tenv valuation (proj1 Hfits)). }
  destruct (extension_fits_kept _ _ _ Hknows1
              (scoped_extension_fits tenv stenv Hscoped_fits
                 alternative_scoped))
    as (_ & Hextension & Hkinded).
  destruct (meet_env_add_env_extension_strict_sound_for mh statics valuation_s
              menv1 _ added Hkeeps Hsound Hfits1 Hextension Hkinded Hstatics
              Hsatisfied1 Htrue Hadd)
    as (menv2 & -> & valuation' & Hextended' & Hsatisfied' & Hvalued').
  injection Hreturn as <-.
  exists menv2, valuation'. split; [ reflexivity | ]. split.
  { exact (extended_from_knows _ _ _ _ Hknows1 Hextended'). }
  split; [ exact Hsatisfied' | ].
  intros cut_after Hvalued. apply Hvalued'. apply (Hvalued01 valuation_s).
  exact (equations_valued_extended tenv cut_after valuation valuation_s
           (proj1 Hfits) Hextended Hvalued).
Qed.

End Alternative.

(* ======================================================================== *)
(* The two sides are joined                                                 *)
(* ======================================================================== *)

(* [env_a] and [env_b] are the environments of two alternatives of a
   meet in [initial_env], the second being given the names to avoid of
   the first. *)
Section Joined.

Variable join : join_scoped_envs.
Hypothesis Hjoin : join_scoped_envs_sound join.
Variable choices : meet_choices.
Hypothesis Hchoice : define_again choices = false.
Variable mh : meet_expanded_head.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.

Variable initial_env base_a env_a base_b env_b : Meet_env.t.

Local Abbreviation tenv := (Meet_env.typing_env initial_env).
Local Abbreviation tenv_a := (Meet_env.typing_env env_a).
Local Abbreviation tenv_b := (Meet_env.typing_env env_b).
Local Abbreviation level_a :=
  (Typing_env.One_level.level (Typing_env.current_level tenv_a)).
Local Abbreviation level_b :=
  (Typing_env.One_level.level (Typing_env.current_level tenv_b)).
Local Abbreviation joining :=
  (Meet_env.with_names_to_avoid initial_env (Meet_env.names_to_avoid env_b)).

Hypothesis Hfits : typing_env_fits tenv.
Hypothesis Hbase_a :
  Meet_env.typing_env base_a = Typing_env.increment_scope tenv.
Hypothesis Hbase_a_avoid :
  ∀ x : variable,
    among x (Meet_env.names_to_avoid initial_env) = true ->
    among x (Meet_env.names_to_avoid base_a) = true.
Hypothesis Hextends_a : meet_env_extends base_a env_a.
Hypothesis Hfits_a : typing_env_fits tenv_a.
Hypothesis Hnames_a : new_names_avoided base_a env_a.
Hypothesis Hbase_b :
  Meet_env.typing_env base_b = Typing_env.increment_scope tenv.
Hypothesis Hbase_b_avoid :
  ∀ x : variable,
    among x (Meet_env.names_to_avoid env_a) = true ->
    among x (Meet_env.names_to_avoid base_b) = true.
Hypothesis Hextends_b : meet_env_extends base_b env_b.
Hypothesis Hfits_b : typing_env_fits tenv_b.
Hypothesis Hnames_b : new_names_avoided base_b env_b.

Lemma joined_scoped_a : tenv_extends (Typing_env.increment_scope tenv) tenv_a.
Proof.
  exact (alternative_scoped initial_env base_a env_a Hbase_a Hextends_a).
Qed.

Lemma joined_scoped_b : tenv_extends (Typing_env.increment_scope tenv) tenv_b.
Proof.
  exact (alternative_scoped initial_env base_b env_b Hbase_b Hextends_b).
Qed.

Lemma joined_avoid_b_initial (x : variable) :
  among x (Meet_env.names_to_avoid initial_env) = true ->
  among x (Meet_env.names_to_avoid base_b) = true.
Proof.
  intros Hx. apply Hbase_b_avoid.
  exact (meet_env_extends_avoid _ _ Hextends_a x (Hbase_a_avoid x Hx)).
Qed.

(* A variable that both alternatives know was known before. *)
Lemma alternatives_disjoint (x : variable) :
  Typing_env.mem None tenv_a (Name_var x) = true ->
  Typing_env.mem None tenv_b (Name_var x) = true ->
  Typing_env.mem None tenv (Name_var x) = true.
Proof.
  intros Ha Hb.
  destruct (Typing_env.mem None tenv (Name_var x)) eqn:Hx; [ reflexivity | ].
  exfalso.
  destruct (alternative_new_avoided initial_env base_a env_a Hbase_a Hnames_a
              x Ha Hx) as (_ & Hamong & _).
  destruct (alternative_new_avoided initial_env base_b env_b Hbase_b Hnames_b
              x Hb Hx) as [Hnot _].
  rewrite (Hbase_b_avoid x Hamong) in Hnot. discriminate Hnot.
Qed.

(* No simple is at two kinds in the two. *)
Lemma alternatives_agree (s : simple) (kind kind' : flambda_kind) :
  simple_of_kind tenv_a s kind -> simple_of_kind tenv_b s kind' ->
  kind = kind'.
Proof.
  intros Ha Hb.
  destruct (Typing_env.mem_simple None tenv s) eqn:Hknown.
  - exact (simple_of_kind_unique tenv s kind kind'
             (scoped_env_of_kind tenv tenv_a s kind
                (scoped_scoped_env_of tenv tenv_a Hfits_a joined_scoped_a)
                Hknown Ha)
             (scoped_env_of_kind tenv tenv_b s kind'
                (scoped_scoped_env_of tenv tenv_b Hfits_b joined_scoped_b)
                Hknown Hb)).
  - exfalso. destruct s as [ x | sym | c ]; [ | | discriminate Hknown ].
    + cbn [Typing_env.mem_simple] in Hknown.
      rewrite (alternatives_disjoint x
                 (simple_of_kind_mem tenv_a (Name_var x) kind (proj1 Hfits_a)
                    Ha)
                 (simple_of_kind_mem tenv_b (Name_var x) kind' (proj1 Hfits_b)
                    Hb)) in Hknown.
      discriminate Hknown.
    + cbn [Typing_env.mem_simple] in Hknown.
      rewrite <- (scoped_symbol_known tenv tenv_a joined_scoped_a sym)
        in Hknown.
      rewrite (simple_of_kind_mem tenv_a (Name_symbol sym) kind
                 (proj1 Hfits_a) Ha) in Hknown.
      discriminate Hknown.
Qed.

Lemma joined_envs_agree : scoped_envs_agree [tenv_a; tenv_b].
Proof.
  intros scoped_env scoped_env' s kind kind' Hin Hin' Hs Hs'.
  destruct Hin as [ <- | [ <- | [] ] ]; destruct Hin' as [ <- | [ <- | [] ] ].
  - exact (simple_of_kind_unique _ s kind kind' Hs Hs').
  - exact (alternatives_agree s kind kind' Hs Hs').
  - symmetry. exact (alternatives_agree s kind' kind Hs' Hs).
  - exact (simple_of_kind_unique _ s kind kind' Hs Hs').
Qed.

Lemma joined_envs_of (scoped_env : Typing_env.t) :
  scoped_env ∈ [tenv_a; tenv_b] -> scoped_env_of tenv scoped_env.
Proof.
  intros [ <- | [ <- | [] ] ].
  - exact (scoped_scoped_env_of tenv tenv_a Hfits_a joined_scoped_a).
  - exact (scoped_scoped_env_of tenv tenv_b Hfits_b joined_scoped_b).
Qed.

Section After_join.

Variable result_env menv1 menv2 : Meet_env.t.

Hypothesis Hjoined :
  join mh (Typing_env.current_scope tenv) joining tenv [tenv_a; tenv_b]
  = Some result_env.
Hypothesis Hdefined_a :
  add_defined_vars choices result_env level_a = Some menv1.
Hypothesis Hdefined_b : add_defined_vars choices menv1 level_b = Some menv2.

Local Abbreviation rtenv := (Meet_env.typing_env result_env).

(* What the join's contract gives. *)
Lemma joined_extends : meet_env_extends joining result_env.
Proof.
  exact (proj1 (proj2 (Hjoin mh joining result_env [tenv_a; tenv_b] Hsound
                         Hkeeps Hfits joined_envs_of joined_envs_agree
                         Hjoined))).
Qed.

Lemma joined_fits : typing_env_fits rtenv.
Proof.
  exact (proj1 (proj2 (proj2
           (Hjoin mh joining result_env [tenv_a; tenv_b] Hsound Hkeeps Hfits
              joined_envs_of joined_envs_agree Hjoined)))).
Qed.

Lemma joined_new (x : variable) :
  Typing_env.mem None rtenv (Name_var x) = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  Typing_env.mem None tenv_a (Name_var x) = false ->
  Typing_env.mem None tenv_b (Name_var x) = false ->
  among x (Meet_env.names_to_avoid env_b) = false
  ∧ among x (Meet_env.names_to_avoid result_env) = true.
Proof.
  intros Hknown Hunknown Ha Hb.
  apply (proj1 (proj2 (proj2 (proj2
           (Hjoin mh joining result_env [tenv_a; tenv_b] Hsound Hkeeps Hfits
              joined_envs_of joined_envs_agree Hjoined)))) x Hknown Hunknown).
  intros scoped_env [ <- | [ <- | [] ] ]; assumption.
Qed.

Lemma joined_kinds (scoped_env : Typing_env.t) (s : simple)
    (kind : flambda_kind) :
  scoped_env ∈ [tenv_a; tenv_b] ->
  Typing_env.mem_simple None rtenv s = true ->
  simple_of_kind scoped_env s kind ->
  simple_of_kind rtenv s kind.
Proof.
  exact (proj2 (proj2 (proj2 (proj2 (proj2
           (Hjoin mh joining result_env [tenv_a; tenv_b] Hsound Hkeeps Hfits
              joined_envs_of joined_envs_agree Hjoined))))) scoped_env s kind).
Qed.

(* A variable of the result that the first environment does not know
   may not occur in terms. *)
Lemma joined_not_in_terms (x : variable) :
  Typing_env.mem None tenv (Name_var x) = false ->
  Typing_env.mem (Some Name_mode.normal) rtenv (Name_var x) = false.
Proof.
  intros Hunknown.
  destruct (Typing_env.mem None rtenv (Name_var x)) eqn:Hknown;
    [ | exact (unknown_not_in_terms rtenv x Hknown) ].
  exact (proj1 (proj2 (proj2 (proj2 (proj2
           (Hjoin mh joining result_env [tenv_a; tenv_b] Hsound Hkeeps Hfits
              joined_envs_of joined_envs_agree Hjoined))))) x Hknown
           Hunknown).
Qed.

Lemma joined_sound (scoped_env : Typing_env.t) (statics : static_env)
    (valuation : env) :
  scoped_env ∈ [tenv_a; tenv_b] ->
  statics_valid statics ->
  typing_env_satisfied_strictly statics valuation tenv ->
  typing_env_satisfied_strictly statics valuation scoped_env ->
  equations_valued scoped_env (Typing_env.current_scope tenv) valuation ->
  ∃ valuation' : env,
    extended_from scoped_env valuation valuation'
    ∧ typing_env_satisfied_strictly statics valuation' rtenv
    ∧ keeps_equations_valued tenv rtenv valuation valuation'.
Proof.
  exact (proj1 (Hjoin mh joining result_env [tenv_a; tenv_b] Hsound Hkeeps
                  Hfits joined_envs_of joined_envs_agree Hjoined)
           scoped_env statics valuation).
Qed.

Lemma joined_def_a :
  defined_from (Typing_env_level.find_kind level_a)
    (Typing_env_level.variables_in_order level_a) result_env menv1.
Proof.
  exact (add_defined_vars_defined choices Hchoice result_env menv1 level_a
           joined_fits
           (scoped_level_indexed tenv tenv_a Hfits_a joined_scoped_a)
           Hdefined_a).
Qed.

Lemma joined_def_b :
  defined_from (Typing_env_level.find_kind level_b)
    (Typing_env_level.variables_in_order level_b) menv1 menv2.
Proof.
  exact (add_defined_vars_defined choices Hchoice menv1 menv2 level_b
           (proj1 (proj2 joined_def_a))
           (scoped_level_indexed tenv tenv_b Hfits_b joined_scoped_b)
           Hdefined_b).
Qed.

Lemma joined_knows_initial : knows_at_kinds tenv rtenv.
Proof. exact (meet_env_extends_knows _ _ joined_extends). Qed.

Lemma joined_knows_a1 : knows_at_kinds tenv_a (Meet_env.typing_env menv1).
Proof.
  apply (scoped_knows_after tenv tenv_a Hfits Hfits_a joined_scoped_a
           result_env menv1 joined_knows_initial); [ | exact joined_def_a ].
  intros s kind Hknown Hs.
  exact (joined_kinds tenv_a s kind (or_introl eq_refl) Hknown Hs).
Qed.

Lemma joined_knows_a : knows_at_kinds tenv_a (Meet_env.typing_env menv2).
Proof.
  exact (knows_at_kinds_trans _ _ _ joined_knows_a1
           (meet_env_extends_knows _ _ (proj1 joined_def_b))).
Qed.

Lemma joined_knows_b : knows_at_kinds tenv_b (Meet_env.typing_env menv2).
Proof.
  destruct joined_def_a as (Hextends1 & _ & _ & Hmem1 & Hkind1 & _).
  assert (Htenv1 := meet_env_extends_tenv _ _ Hextends1).
  apply (scoped_knows_after tenv tenv_b Hfits Hfits_b joined_scoped_b menv1
           menv2
           (knows_at_kinds_trans _ _ _ joined_knows_initial
              (tenv_extends_knows _ _ Htenv1)));
    [ | exact joined_def_b ].
  intros s kind Hknown1 Hs.
  destruct (Typing_env.mem_simple None rtenv s) eqn:Hknown0.
  { exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv1)))) s kind
             (joined_kinds tenv_b s kind (or_intror (or_introl eq_refl))
                Hknown0 Hs)). }
  destruct s as [ x | sym | c ]; [ | | discriminate Hknown0 ].
  - cbn [Typing_env.mem_simple] in Hknown0, Hknown1.
    destruct (Hmem1 x Hknown1) as [ Hold | [Hin Hsome] ].
    { rewrite Hold in Hknown0. discriminate Hknown0. }
    destruct (Typing_env_level.find_kind level_a x) as [ k | ] eqn:Hk;
      [ | exfalso; apply Hsome; reflexivity ].
    assert (Hsame : k = kind).
    { exact (alternatives_agree (Var x) k kind
               (scoped_defined_kind tenv tenv_a Hfits_a joined_scoped_a x k Hk)
               Hs). }
    subst k. exact (Hkind1 x kind Hin Hk Hknown0).
  - exfalso. cbn [Typing_env.mem_simple] in Hknown0, Hknown1.
    unfold Typing_env.mem in Hknown0, Hknown1.
    rewrite (proj1 (proj2 (proj2 (proj2 Htenv1)))) in Hknown1.
    rewrite Hknown1 in Hknown0. discriminate Hknown0.
Qed.

Lemma join_return_env_keeps :
  meet_env_extends initial_env menv2
  ∧ typing_env_fits (Meet_env.typing_env menv2)
  ∧ new_names_avoided initial_env menv2.
Proof.
  destruct joined_def_a as (Hextends1 & Hfits1 & Havoid1 & Hmem1 & _).
  destruct joined_def_b as (Hextends2 & Hfits2 & Havoid2 & Hmem2 & _).
  assert (Hinitial : meet_env_extends initial_env joining).
  { repeat split; try reflexivity.
    - intros s kind Hs. exact Hs.
    - intros x Hx.
      exact (meet_env_extends_avoid _ _ Hextends_b x
               (joined_avoid_b_initial x Hx)).
    - apply keeps_names_refl. }
  split.
  { exact (meet_env_extends_trans _ _ _ Hinitial
             (meet_env_extends_trans _ _ _ joined_extends
                (meet_env_extends_trans _ _ _ Hextends1 Hextends2))). }
  split; [ exact Hfits2 | ].
  intros x Hknown Hunknown.
  assert (Hfinal :
    among x (Meet_env.names_to_avoid env_b) = true ->
    among x (Meet_env.names_to_avoid menv2) = true).
  { intros Hb. rewrite Havoid2, Havoid1.
    exact (meet_env_extends_avoid _ _ joined_extends x Hb). }
  (* The variable may not occur in terms: not in the result of the
     join, and the variables defined after it are defined in the name
     mode of types. *)
  assert (Hterms :
    Typing_env.mem (Some Name_mode.normal) (Meet_env.typing_env menv2)
      (Name_var x)
    = false).
  { apply (add_defined_vars_not_in_terms choices menv1 menv2 level_b x
             (scoped_level_indexed tenv tenv_b Hfits_b joined_scoped_b)
             Hdefined_b).
    apply (add_defined_vars_not_in_terms choices result_env menv1 level_a x
             (scoped_level_indexed tenv tenv_a Hfits_a joined_scoped_a)
             Hdefined_a).
    exact (joined_not_in_terms x Hunknown). }
  destruct (Typing_env.mem None tenv_a (Name_var x)) eqn:Ha.
  { destruct (alternative_new_avoided initial_env base_a env_a Hbase_a
                Hnames_a x Ha Hunknown) as (Hnot & Hamong & _).
    split; [ | split; [ | exact Hterms ] ].
    - destruct (among x (Meet_env.names_to_avoid initial_env)) eqn:Hi;
        [ | reflexivity ].
      rewrite (Hbase_a_avoid x Hi) in Hnot. discriminate Hnot.
    - apply Hfinal.
      exact (meet_env_extends_avoid _ _ Hextends_b x
               (Hbase_b_avoid x Hamong)). }
  destruct (Typing_env.mem None tenv_b (Name_var x)) eqn:Hb.
  { destruct (alternative_new_avoided initial_env base_b env_b Hbase_b
                Hnames_b x Hb Hunknown) as (Hnot & Hamong & _).
    split; [ | split; [ exact (Hfinal Hamong) | exact Hterms ] ].
    destruct (among x (Meet_env.names_to_avoid initial_env)) eqn:Hi;
      [ | reflexivity ].
    rewrite (joined_avoid_b_initial x Hi) in Hnot. discriminate Hnot. }
  assert (Hresult : Typing_env.mem None rtenv (Name_var x) = true).
  { destruct (Hmem2 x Hknown) as [ Hold1 | [Hin Hsome] ].
    - destruct (Hmem1 x Hold1) as [ Hold0 | [Hin Hsome] ]; [ exact Hold0 | ].
      exfalso.
      destruct (Typing_env_level.find_kind level_a x) as [ k | ] eqn:Hk;
        [ | apply Hsome; reflexivity ].
      rewrite (scoped_in_order_known tenv tenv_a Hfits_a joined_scoped_a x k
                 Hk) in Ha.
      discriminate Ha.
    - exfalso.
      destruct (Typing_env_level.find_kind level_b x) as [ k | ] eqn:Hk;
        [ | apply Hsome; reflexivity ].
      rewrite (scoped_in_order_known tenv tenv_b Hfits_b joined_scoped_b x k
                 Hk) in Hb.
      discriminate Hb. }
  destruct (joined_new x Hresult Hunknown Ha Hb) as [Hnot Hamong].
  split; [ | split; [ | exact Hterms ] ].
  - destruct (among x (Meet_env.names_to_avoid initial_env)) eqn:Hi;
      [ | reflexivity ].
    rewrite (meet_env_extends_avoid _ _ Hextends_b x
               (joined_avoid_b_initial x Hi)) in Hnot.
    discriminate Hnot.
  - rewrite Havoid2, Havoid1. exact Hamong.
Qed.

(* No value of the alternative [scoped_x] is lost. *)
Lemma join_return_env_sound (statics : static_env)
    (valuation valuation_x : env) (scoped_x : Typing_env.t) :
  scoped_x = tenv_a ∨ scoped_x = tenv_b ->
  statics_valid statics ->
  typing_env_satisfied_strictly statics valuation tenv ->
  extended_from tenv valuation valuation_x ->
  typing_env_satisfied_strictly statics valuation_x scoped_x ->
  keeps_equations_valued (Typing_env.increment_scope tenv) scoped_x valuation
    valuation_x ->
  ∃ valuation' : env,
    extended_from scoped_x valuation_x valuation'
    ∧ typing_env_satisfied_strictly statics valuation'
        (Meet_env.typing_env menv2)
    ∧ keeps_equations_valued tenv (Meet_env.typing_env menv2) valuation
        valuation'.
Proof.
  intros Hwhich Hstatics Hsatisfied Hextended Hsatisfied_x Hvalued_x.
  assert (Hin : scoped_x ∈ [tenv_a; tenv_b]).
  { destruct Hwhich as [ -> | -> ]; [ left | right; left ]; reflexivity. }
  destruct (joined_sound scoped_x statics valuation_x Hin Hstatics
              (satisfied_strictly_extended statics tenv valuation valuation_x
                 (proj1 (proj1 Hfits)) Hextended Hsatisfied)
              Hsatisfied_x
              (Hvalued_x _ (increment_scope_valued tenv valuation
                              (proj1 Hfits))))
    as (valuation_j & Hextended_j & Hsatisfied_j & Hvalued_j).
  set (keep := fun x : variable =>
    Typing_env.mem None rtenv (Name_var x)
    || Typing_env.mem None scoped_x (Name_var x)).
  assert (Hrestricted :
    extended_from rtenv valuation_j (restrict_valuation keep valuation_j)).
  { intros x Hx. rewrite lookup_restrict_valuation. unfold keep.
    rewrite Hx. reflexivity. }
  assert (Hextended' :
    extended_from scoped_x valuation_x (restrict_valuation keep valuation_j)).
  { intros x Hx. rewrite lookup_restrict_valuation. unfold keep.
    rewrite Hx, orb_true_r. exact (Hextended_j x Hx). }
  assert (Hsatisfied0 :=
    satisfied_strictly_extended statics rtenv valuation_j
      (restrict_valuation keep valuation_j) (proj1 (proj1 joined_fits))
      Hrestricted Hsatisfied_j).
  destruct joined_def_a
    as (Hextends1 & Hfits1 & _ & _ & _ & _ & Hsatisfied1 & Hvalued1).
  destruct joined_def_b
    as (Hextends2 & Hfits2 & _ & _ & _ & _ & Hsatisfied2 & Hvalued2).
  exists (restrict_valuation keep valuation_j).
  split; [ exact Hextended' | ]. split.
  - apply (Hsatisfied2 statics).
    + apply (Hsatisfied1 statics _ Hsatisfied0).
      intros y k v _ Hk Hunknown Hv.
      assert (Hy_a := scoped_in_order_known tenv tenv_a Hfits_a
                        joined_scoped_a y k Hk).
      destruct Hwhich as [ Hwhich | Hwhich ]; subst scoped_x.
      * rewrite (Hextended' y Hy_a) in Hv.
        exact (scoped_defined_value_kind tenv tenv_a Hfits_a joined_scoped_a
                 statics valuation_x y k v Hsatisfied_x Hk Hv).
      * exfalso. rewrite lookup_restrict_valuation in Hv. unfold keep in Hv.
        rewrite Hunknown in Hv.
        destruct (Typing_env.mem None tenv_b (Name_var y)) eqn:Hy_b;
          [ | discriminate Hv ].
        rewrite (proj1 joined_knows_initial _
                   (alternatives_disjoint y Hy_a Hy_b)) in Hunknown.
        discriminate Hunknown.
    + intros y k v _ Hk Hunknown Hv.
      assert (Hy_b := scoped_in_order_known tenv tenv_b Hfits_b
                        joined_scoped_b y k Hk).
      assert (Hunknown0 : Typing_env.mem None rtenv (Name_var y) = false).
      { destruct (Typing_env.mem None rtenv (Name_var y)) eqn:Hy0;
          [ | reflexivity ].
        rewrite (tenv_extends_mem _ _ _ (meet_env_extends_tenv _ _ Hextends1)
                   Hy0) in Hunknown.
        discriminate Hunknown. }
      destruct Hwhich as [ Hwhich | Hwhich ]; subst scoped_x.
      * exfalso. rewrite lookup_restrict_valuation in Hv. unfold keep in Hv.
        rewrite Hunknown0 in Hv.
        destruct (Typing_env.mem None tenv_a (Name_var y)) eqn:Hy_a;
          [ | discriminate Hv ].
        rewrite (proj1 joined_knows_initial _
                   (alternatives_disjoint y Hy_a Hy_b)) in Hunknown0.
        discriminate Hunknown0.
      * rewrite (Hextended' y Hy_b) in Hv.
        exact (scoped_defined_value_kind tenv tenv_b Hfits_b joined_scoped_b
                 statics valuation_x y k v Hsatisfied_x Hk Hv).
  - intros cut_after Hvalued. apply (Hvalued2 _). apply (Hvalued1 _).
    apply (equations_valued_extended rtenv cut_after valuation_j _
             (proj1 joined_fits) Hrestricted).
    apply Hvalued_j.
    exact (equations_valued_extended tenv cut_after valuation valuation_x
             (proj1 Hfits) Hextended Hvalued).
Qed.

End After_join.

End Joined.

(* ======================================================================== *)
(* The meet of two things with two sides                                    *)
(* ======================================================================== *)

Lemma increment_scope_knows (tenv : Typing_env.t) :
  knows_at_kinds tenv (Typing_env.increment_scope tenv).
Proof.
  split.
  - intros n Hn. exact Hn.
  - intros s kind Hs. apply increment_scope_simple_of_kind. exact Hs.
Qed.

(* The answer for a pair is the first input, the second, or the new
   thing. *)
Lemma extract_combine {X Y Z : Type} (rx : meet_return_value X)
    (ry : meet_return_value Y) (f : unit -> Z) (z1 z2 : Z) :
  extract_value (combine_meet_return_values rx ry f) z1 z2 = z1
  ∨ extract_value (combine_meet_return_values rx ry f) z1 z2 = z2
  ∨ extract_value (combine_meet_return_values rx ry f) z1 z2 = f tt.
Proof. destruct rx, ry; cbn; auto. Qed.

Section Disjunction.

Context {A B : Type}.

Variable fits_a : Typing_env.t -> A -> Prop.
Variable satisfied_a : static_env -> env -> A -> Prop.
Variable fits_b : Typing_env.t -> B -> Prop.
Variable satisfied_b : static_env -> env -> B -> Prop.

Hypothesis Hstable_a : part_stable fits_a satisfied_a.
Hypothesis Hstable_b : part_stable fits_b satisfied_b.

Local Abbreviation fits := (disjunction_fits fits_a fits_b).
Local Abbreviation satisfied := (disjunction_satisfied satisfied_a satisfied_b).

Lemma disjunction_fits_kept (tenv tenv' : Typing_env.t)
    (x : A * B * variant_extensions) :
  fits_by_kinds fits_a ->
  fits_by_kinds fits_b ->
  knows_at_kinds tenv tenv' ->
  fits tenv x -> fits tenv' x.
Proof.
  intros Hkinds_a Hkinds_b Hknows. destruct x as [[a b] extensions].
  intros (Ha & Hb & Hextension_a & Hextension_b).
  split; [ exact (Hkinds_a tenv tenv' a Hknows Ha) | ].
  split; [ exact (Hkinds_b tenv tenv' b Hknows Hb) | ].
  split.
  - exact (optional_extension_fits_kept _ _ _ Hknows Hextension_a).
  - exact (optional_extension_fits_kept _ _ _ Hknows Hextension_b).
Qed.

Lemma disjunction_stable : part_stable fits satisfied.
Proof.
  split.
  - intros menv menv' [[a b] extensions] Hextends
      (Ha & Hb & Hextension_a & Hextension_b).
    assert (Hknows := meet_env_extends_knows _ _ Hextends).
    split; [ exact (proj1 Hstable_a menv menv' a Hextends Ha) | ].
    split; [ exact (proj1 Hstable_b menv menv' b Hextends Hb) | ].
    split.
    + exact (optional_extension_fits_kept _ _ _ Hknows Hextension_a).
    + exact (optional_extension_fits_kept _ _ _ Hknows Hextension_b).
  - intros statics tenv valuation valuation' [[a b] extensions]
      (Ha & Hb & Hextension_a & Hextension_b) Hextended
      [ [Hsatisfied Htrue] | [Hsatisfied Htrue] ]; [ left | right ].
    + split.
      * exact (proj2 Hstable_a statics tenv valuation valuation' a Ha
                 Hextended Hsatisfied).
      * exact (optional_extension_satisfied_extended statics tenv valuation
                 valuation' _ Hextension_a Hextended Htrue).
    + split.
      * exact (proj2 Hstable_b statics tenv valuation valuation' b Hb
                 Hextended Hsatisfied).
      * exact (optional_extension_satisfied_extended statics tenv valuation
                 valuation' _ Hextension_b Hextended Htrue).
Qed.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh : meet_expanded_head.
Variable meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A).
Variable meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B).
Variable bottom_a : unit -> A.
Variable bottom_b : unit -> B.

Hypothesis Hjoin : join_scoped_envs_sound join.
Hypothesis Hchoice : define_again choices = false.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.
Hypothesis Hkinds_a : fits_by_kinds fits_a.
Hypothesis Hkinds_b : fits_by_kinds fits_b.
Hypothesis Hmeet_a : part_meet_ok fits_a satisfied_a meet_a.
Hypothesis Hmeet_b : part_meet_ok fits_b satisfied_b meet_b.
Hypothesis Hexclusive :
  ∀ (statics : static_env) (valuation : env) (a : A) (b : B),
    satisfied_a statics valuation a -> satisfied_b statics valuation b -> False.
Hypothesis Hbottom_a : ∀ tenv : Typing_env.t, fits_a tenv (bottom_a tt).
Hypothesis Hbottom_b : ∀ tenv : Typing_env.t, fits_b tenv (bottom_b tt).

Variable menv : Meet_env.t.
Variable a1 a2 : A.
Variable b1 b2 : B.
Variable e1 e2 : variant_extensions.

Local Abbreviation tenv := (Meet_env.typing_env menv).
Local Abbreviation x1 := (a1, b1, e1).
Local Abbreviation x2 := (a2, b2, e2).

Hypothesis Hfits : typing_env_fits tenv.
Hypothesis Hx1 : fits tenv x1.
Hypothesis Hx2 : fits tenv x2.

(* What an environment that is the environment of the meet one scope
   further has. *)
Lemma base_facts (base : Meet_env.t) :
  Meet_env.typing_env base = Typing_env.increment_scope tenv ->
  typing_env_fits (Meet_env.typing_env base)
  ∧ fits (Meet_env.typing_env base) x1
  ∧ fits (Meet_env.typing_env base) x2
  ∧ (∀ (statics : static_env) (valuation : env),
       typing_env_satisfied_strictly statics valuation tenv ->
       typing_env_satisfied_strictly statics valuation
         (Meet_env.typing_env base))
  ∧ (∀ valuation valuation' : env,
       extended_from (Meet_env.typing_env base) valuation valuation' ->
       extended_from tenv valuation valuation').
Proof.
  intros ->.
  split; [ exact (increment_scope_fits tenv Hfits) | ].
  split.
  { exact (disjunction_fits_kept _ _ x1 Hkinds_a Hkinds_b
             (increment_scope_knows tenv) Hx1). }
  split.
  { exact (disjunction_fits_kept _ _ x2 Hkinds_a Hkinds_b
             (increment_scope_knows tenv) Hx2). }
  split.
  - intros statics valuation Hsatisfied.
    apply (increment_scope_satisfied (fun _ => False)). exact Hsatisfied.
  - intros valuation valuation' Hextended. exact Hextended.
Qed.

(* One call. *)
Lemma meet_disjunction_call
    (result : Meet_result.t (A * B * variant_extensions)) :
  meet_disjunction join choices mh meet_a meet_b bottom_a bottom_b menv a1 b1
    e1 a2 b2 e2
  = Some result ->
  meet_call_ok fits satisfied menv x1 x2 result.
Proof.
  intros Hmeet. rewrite meet_disjunction_eq in Hmeet. cbv zeta in Hmeet.
  set (base_a := snd (Meet_env.enter_scope menv)) in *.
  destruct (add_optional_extension mh (Or_bottom.Ok base_a)
              (extension_when_a e1)) as [ env_a1 | ] eqn:Ha1;
    [ | discriminate Hmeet ].
  cbv beta iota in Hmeet.
  destruct (add_optional_extension mh env_a1 (extension_when_a e2))
    as [ env_a2 | ] eqn:Ha2; [ | discriminate Hmeet ].
  cbv beta iota in Hmeet.
  destruct (meet_side meet_a env_a2 a1 a2) as [ a_result | ] eqn:Ha3;
    [ | discriminate Hmeet ].
  cbv beta iota in Hmeet.
  set (base_b :=
    match a_result with
    | Meet_result.Bottom _ => base_a
    | Meet_result.Ok _ env_a =>
      Meet_env.with_names_to_avoid base_a (Meet_env.names_to_avoid env_a)
    end) in *.
  destruct (add_optional_extension mh (Or_bottom.Ok base_b)
              (extension_when_b e1)) as [ env_b1 | ] eqn:Hb1;
    [ | discriminate Hmeet ].
  cbv beta iota in Hmeet.
  destruct (add_optional_extension mh env_b1 (extension_when_b e2))
    as [ env_b2 | ] eqn:Hb2; [ | discriminate Hmeet ].
  cbv beta iota in Hmeet.
  destruct (meet_side meet_b env_b2 b1 b2) as [ b_result | ] eqn:Hb3;
    [ | discriminate Hmeet ].
  cbv beta iota in Hmeet.
  assert (Hbase_a :
    Meet_env.typing_env base_a = Typing_env.increment_scope tenv).
  { reflexivity. }
  assert (Hbase_b :
    Meet_env.typing_env base_b = Typing_env.increment_scope tenv).
  { unfold base_b. destruct a_result; reflexivity. }
  destruct (base_facts base_a Hbase_a)
    as (Hfits_base_a & (Ha1_fits & _ & Hea1 & _) & (Ha2_fits & _ & Hea2 & _)
        & Hsatisfied_base_a & Hextended_base_a).
  destruct (base_facts base_b Hbase_b)
    as (Hfits_base_b & (_ & Hb1_fits & _ & Heb1) & (_ & Hb2_fits & _ & Heb2)
        & Hsatisfied_base_b & Hextended_base_b).
  (* What the two sides keep. *)
  assert (Hside_a :
    ∀ (r : meet_return_value A) (env_a : Meet_env.t),
      a_result = Meet_result.Ok r env_a ->
      meet_env_extends base_a env_a
      ∧ typing_env_fits (Meet_env.typing_env env_a)
      ∧ fits_a (Meet_env.typing_env env_a) (extract_value r a1 a2)
      ∧ new_names_avoided base_a env_a).
  { intros r env_a Heq. rewrite Heq in Ha3.
    exact (side_keeps fits_a satisfied_a mh meet_a Hkeeps Hstable_a Hmeet_a
             base_a (extension_when_a e1) (extension_when_a e2) a1 a2
             Hfits_base_a Hea1 Hea2 Ha1_fits Ha2_fits env_a1 env_a2 r env_a
             Ha1 Ha2 Ha3). }
  assert (Hside_b :
    ∀ (r : meet_return_value B) (env_b : Meet_env.t),
      b_result = Meet_result.Ok r env_b ->
      meet_env_extends base_b env_b
      ∧ typing_env_fits (Meet_env.typing_env env_b)
      ∧ fits_b (Meet_env.typing_env env_b) (extract_value r b1 b2)
      ∧ new_names_avoided base_b env_b).
  { intros r env_b Heq. rewrite Heq in Hb3.
    exact (side_keeps fits_b satisfied_b mh meet_b Hkeeps Hstable_b Hmeet_b
             base_b (extension_when_b e1) (extension_when_b e2) b1 b2
             Hfits_base_b Heb1 Heb2 Hb1_fits Hb2_fits env_b1 env_b2 r env_b
             Hb1 Hb2 Hb3). }
  (* The two sides lose no value. *)
  assert (Hlost_a :
    ∀ (statics : static_env) (valuation : env),
      statics_valid statics ->
      typing_env_satisfied_strictly statics valuation tenv ->
      satisfied_a statics valuation a1
      ∧ optional_extension_satisfied statics valuation (extension_when_a e1)
      ->
      satisfied_a statics valuation a2
      ∧ optional_extension_satisfied statics valuation (extension_when_a e2)
      ->
      ∃ (r : meet_return_value A) (env_a : Meet_env.t) (valuation' : env),
        a_result = Meet_result.Ok r env_a
        ∧ extended_from tenv valuation valuation'
        ∧ typing_env_satisfied_strictly statics valuation'
            (Meet_env.typing_env env_a)
        ∧ satisfied_a statics valuation' (extract_value r a1 a2)
        ∧ keeps_equations_valued (Meet_env.typing_env base_a)
            (Meet_env.typing_env env_a) valuation valuation').
  { intros statics valuation Hstatics Hsatisfied [Hv1 Htrue1] [Hv2 Htrue2].
    destruct (side_sound fits_a satisfied_a mh meet_a Hkeeps Hsound Hstable_a
                Hmeet_a base_a (extension_when_a e1) (extension_when_a e2) a1
                a2 Hfits_base_a Hea1 Hea2 Ha1_fits Ha2_fits statics valuation
                env_a1 env_a2 a_result Hstatics
                (Hsatisfied_base_a statics valuation Hsatisfied) Htrue1 Htrue2
                Hv1 Hv2 Ha1 Ha2 Ha3)
      as (r & env_a & valuation' & Heq & Hextended & Hrest).
    exists r, env_a, valuation'. split; [ exact Heq | ].
    split; [ exact (Hextended_base_a _ _ Hextended) | exact Hrest ]. }
  assert (Hlost_b :
    ∀ (statics : static_env) (valuation : env),
      statics_valid statics ->
      typing_env_satisfied_strictly statics valuation tenv ->
      satisfied_b statics valuation b1
      ∧ optional_extension_satisfied statics valuation (extension_when_b e1)
      ->
      satisfied_b statics valuation b2
      ∧ optional_extension_satisfied statics valuation (extension_when_b e2)
      ->
      ∃ (r : meet_return_value B) (env_b : Meet_env.t) (valuation' : env),
        b_result = Meet_result.Ok r env_b
        ∧ extended_from tenv valuation valuation'
        ∧ typing_env_satisfied_strictly statics valuation'
            (Meet_env.typing_env env_b)
        ∧ satisfied_b statics valuation' (extract_value r b1 b2)
        ∧ keeps_equations_valued (Meet_env.typing_env base_b)
            (Meet_env.typing_env env_b) valuation valuation').
  { intros statics valuation Hstatics Hsatisfied [Hv1 Htrue1] [Hv2 Htrue2].
    destruct (side_sound fits_b satisfied_b mh meet_b Hkeeps Hsound Hstable_b
                Hmeet_b base_b (extension_when_b e1) (extension_when_b e2) b1
                b2 Hfits_base_b Heb1 Heb2 Hb1_fits Hb2_fits statics valuation
                env_b1 env_b2 b_result Hstatics
                (Hsatisfied_base_b statics valuation Hsatisfied) Htrue1 Htrue2
                Hv1 Hv2 Hb1 Hb2 Hb3)
      as (r & env_b & valuation' & Heq & Hextended & Hrest).
    exists r, env_b, valuation'. split; [ exact Heq | ].
    split; [ exact (Hextended_base_b _ _ Hextended) | exact Hrest ]. }
  (* A value is of one side in both inputs. *)
  assert (Hsides :
    ∀ (statics : static_env) (valuation : env),
      satisfied statics valuation x1 ->
      satisfied statics valuation x2 ->
      ((satisfied_a statics valuation a1
        ∧ optional_extension_satisfied statics valuation (extension_when_a e1))
       ∧ satisfied_a statics valuation a2
         ∧ optional_extension_satisfied statics valuation
             (extension_when_a e2))
      ∨ ((satisfied_b statics valuation b1
          ∧ optional_extension_satisfied statics valuation
              (extension_when_b e1))
         ∧ satisfied_b statics valuation b2
           ∧ optional_extension_satisfied statics valuation
               (extension_when_b e2))).
  { intros statics valuation [ Hv1 | Hv1 ] [ Hv2 | Hv2 ].
    - left. split; assumption.
    - exfalso.
      exact (Hexclusive statics valuation a1 b2 (proj1 Hv1) (proj1 Hv2)).
    - exfalso.
      exact (Hexclusive statics valuation a2 b1 (proj1 Hv2) (proj1 Hv1)).
    - right. split; assumption. }
  clear Ha1 Ha2 Ha3 Hb1 Hb2 Hb3.
  destruct b_result as [ rb0 | rb env_b ];
    destruct a_result as [ ra0 | ra env_a ]; cbv beta iota in Hmeet.
  - (* Both sides are bottom. *)
    injection Hmeet as <-. split; [ intros r menv' Heq; discriminate Heq | ].
    intros statics valuation Hstatics Hsatisfied Hv1 Hv2. exfalso.
    destruct (Hsides statics valuation Hv1 Hv2) as [ [Hs1 Hs2] | [Hs1 Hs2] ].
    + destruct (Hlost_a statics valuation Hstatics Hsatisfied Hs1 Hs2)
        as (r & env_c & valuation' & Heq & _).
      discriminate Heq.
    + destruct (Hlost_b statics valuation Hstatics Hsatisfied Hs1 Hs2)
        as (r & env_c & valuation' & Heq & _).
      discriminate Heq.
  - (* The second side is bottom. *)
    destruct (Hside_a ra env_a eq_refl)
      as (Hextends_a & Hfits_a & Hanswer_a & Hnames_a).
    split.
    + intros r menv' Heq. rewrite Heq in Hmeet.
      destruct (direct_return_keeps choices Hchoice mh Hkeeps menv base_a
                  env_a Hfits Hbase_a (fun x Hx => Hx) Hextends_a Hfits_a
                  Hnames_a _ r menv' Hmeet)
        as (-> & Hextends & Hfits' & Hnames & Hknows).
      split; [ exact Hextends | ]. split; [ exact Hfits' | ].
      split; [ | exact Hnames ].
      destruct (extract_combine ra rb0
                  (fun _ =>
                     (extract_value ra a1 a2, bottom_b tt, No_extensions))
                  x1 x2) as [ -> | [ -> | -> ] ].
      * exact (proj1 disjunction_stable menv menv' x1 Hextends Hx1).
      * exact (proj1 disjunction_stable menv menv' x2 Hextends Hx2).
      * split; [ exact (Hkinds_a _ _ _ Hknows Hanswer_a) | ].
        split; [ apply Hbottom_b | split; exact I ].
    + intros statics valuation Hstatics Hsatisfied Hv1 Hv2.
      destruct (Hsides statics valuation Hv1 Hv2) as [ [Hs1 Hs2] | [Hs1 Hs2] ].
      2:{ destruct (Hlost_b statics valuation Hstatics Hsatisfied Hs1 Hs2)
            as (r & env_c & valuation' & Heq & _).
          discriminate Heq. }
      destruct (Hlost_a statics valuation Hstatics Hsatisfied Hs1 Hs2)
        as (r & env_c & valuation_a & Heq & Hextended_a & Hsatisfied_a
            & Htrue_a & Hvalued_a).
      injection Heq as <- <-.
      destruct (direct_return_sound choices Hchoice mh Hkeeps Hsound menv
                  base_a env_a Hfits Hbase_a Hextends_a Hfits_a statics
                  valuation valuation_a _ result Hstatics Hsatisfied Hextended_a
                  Hsatisfied_a Hvalued_a Hmeet)
        as (result_env & valuation' & -> & Hextended' & Hsatisfied'
            & Hvalued').
      assert (Hextended : extended_from tenv valuation valuation').
      { exact (extended_from_trans _ _ _ _ _
                 (scoped_keeps_names tenv _
                    (alternative_scoped menv base_a env_a Hbase_a Hextends_a))
                 Hextended_a Hextended'). }
      eexists. exists result_env, valuation'. split; [ reflexivity | ].
      split; [ exact Hextended | ]. split; [ exact Hsatisfied' | ].
      split; [ | exact Hvalued' ].
      destruct (extract_combine ra rb0
                  (fun _ =>
                     (extract_value ra a1 a2, bottom_b tt, No_extensions))
                  x1 x2) as [ -> | [ -> | -> ] ].
      * exact (proj2 disjunction_stable statics tenv valuation valuation' x1
                 Hx1 Hextended Hv1).
      * exact (proj2 disjunction_stable statics tenv valuation valuation' x2
                 Hx2 Hextended Hv2).
      * left. split; [ | exact I ].
        exact (proj2 Hstable_a statics _ valuation_a valuation' _ Hanswer_a
                 Hextended' Htrue_a).
  - (* The first side is bottom. *)
    destruct (Hside_b rb env_b eq_refl)
      as (Hextends_b & Hfits_b & Hanswer_b & Hnames_b).
    split.
    + intros r menv' Heq. rewrite Heq in Hmeet.
      destruct (direct_return_keeps choices Hchoice mh Hkeeps menv base_b
                  env_b Hfits Hbase_b (fun x Hx => Hx) Hextends_b Hfits_b
                  Hnames_b _ r menv' Hmeet)
        as (-> & Hextends & Hfits' & Hnames & Hknows).
      split; [ exact Hextends | ]. split; [ exact Hfits' | ].
      split; [ | exact Hnames ].
      destruct (extract_combine ra0 rb
                  (fun _ =>
                     (bottom_a tt, extract_value rb b1 b2, No_extensions))
                  x1 x2) as [ -> | [ -> | -> ] ].
      * exact (proj1 disjunction_stable menv menv' x1 Hextends Hx1).
      * exact (proj1 disjunction_stable menv menv' x2 Hextends Hx2).
      * split; [ apply Hbottom_a | ].
        split; [ exact (Hkinds_b _ _ _ Hknows Hanswer_b) | split; exact I ].
    + intros statics valuation Hstatics Hsatisfied Hv1 Hv2.
      destruct (Hsides statics valuation Hv1 Hv2) as [ [Hs1 Hs2] | [Hs1 Hs2] ].
      { destruct (Hlost_a statics valuation Hstatics Hsatisfied Hs1 Hs2)
          as (r & env_c & valuation' & Heq & _).
        discriminate Heq. }
      destruct (Hlost_b statics valuation Hstatics Hsatisfied Hs1 Hs2)
        as (r & env_c & valuation_b & Heq & Hextended_b & Hsatisfied_b
            & Htrue_b & Hvalued_b).
      injection Heq as <- <-.
      destruct (direct_return_sound choices Hchoice mh Hkeeps Hsound menv
                  base_b env_b Hfits Hbase_b Hextends_b Hfits_b statics
                  valuation valuation_b _ result Hstatics Hsatisfied Hextended_b
                  Hsatisfied_b Hvalued_b Hmeet)
        as (result_env & valuation' & -> & Hextended' & Hsatisfied'
            & Hvalued').
      assert (Hextended : extended_from tenv valuation valuation').
      { exact (extended_from_trans _ _ _ _ _
                 (scoped_keeps_names tenv _
                    (alternative_scoped menv base_b env_b Hbase_b Hextends_b))
                 Hextended_b Hextended'). }
      eexists. exists result_env, valuation'. split; [ reflexivity | ].
      split; [ exact Hextended | ]. split; [ exact Hsatisfied' | ].
      split; [ | exact Hvalued' ].
      destruct (extract_combine ra0 rb
                  (fun _ =>
                     (bottom_a tt, extract_value rb b1 b2, No_extensions))
                  x1 x2) as [ -> | [ -> | -> ] ].
      * exact (proj2 disjunction_stable statics tenv valuation valuation' x1
                 Hx1 Hextended Hv1).
      * exact (proj2 disjunction_stable statics tenv valuation valuation' x2
                 Hx2 Hextended Hv2).
      * right. split; [ | exact I ].
        exact (proj2 Hstable_b statics _ valuation_b valuation' _ Hanswer_b
                 Hextended' Htrue_b).
  - (* Neither side is bottom: the two environments are joined. *)
    destruct (Hside_a ra env_a eq_refl)
      as (Hextends_a & Hfits_a & Hanswer_a & Hnames_a).
    destruct (Hside_b rb env_b eq_refl)
      as (Hextends_b & Hfits_b & Hanswer_b & Hnames_b).
    assert (Hscoped_a :=
      alternative_scoped menv base_a env_a Hbase_a Hextends_a).
    assert (Hscoped_b :=
      alternative_scoped menv base_b env_b Hbase_b Hextends_b).
    unfold join_return, Meet_env.final_typing_env in Hmeet.
    cbv zeta in Hmeet.
    rewrite (scoped_cut tenv _ Hscoped_a), (scoped_cut tenv _ Hscoped_b)
      in Hmeet.
    destruct (join mh (Typing_env.current_scope tenv)
                (Meet_env.with_names_to_avoid menv
                   (Meet_env.names_to_avoid env_b))
                tenv
                [Meet_env.typing_env env_a; Meet_env.typing_env env_b])
      as [ result_env | ] eqn:Hjoined; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    destruct (add_defined_vars choices result_env
                (Typing_env.One_level.level
                   (Typing_env.current_level (Meet_env.typing_env env_a))))
      as [ menv1 | ] eqn:Hdefined_a; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    destruct (add_defined_vars choices menv1
                (Typing_env.One_level.level
                   (Typing_env.current_level (Meet_env.typing_env env_b))))
      as [ menv2 | ] eqn:Hdefined_b; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    set (level_a :=
      Typing_env.One_level.level
        (Typing_env.current_level (Meet_env.typing_env env_a))) in *.
    set (level_b :=
      Typing_env.One_level.level
        (Typing_env.current_level (Meet_env.typing_env env_b))) in *.
    set (extensions :=
      if Typing_env_level.is_empty level_a && Typing_env_level.is_empty level_b
      then No_extensions
      else Ext (extension_of_level level_a) (extension_of_level level_b))
      in *.
    assert (Hknows_a :=
      joined_knows_a join Hjoin choices Hchoice mh Hkeeps Hsound menv base_a
        env_a base_b env_b Hfits Hbase_a Hextends_a Hfits_a Hnames_a Hbase_b
        (fun x Hx => Hx) Hextends_b Hfits_b Hnames_b result_env menv1 menv2
        Hjoined Hdefined_a Hdefined_b).
    assert (Hknows_b :=
      joined_knows_b join Hjoin choices Hchoice mh Hkeeps Hsound menv base_a
        env_a base_b env_b Hfits Hbase_a Hextends_a Hfits_a Hnames_a Hbase_b
        (fun x Hx => Hx) Hextends_b Hfits_b Hnames_b result_env menv1 menv2
        Hjoined Hdefined_a Hdefined_b).
    assert (Hextension_a :=
      scoped_extension_fits tenv _ Hfits_a Hscoped_a).
    assert (Hextension_b :=
      scoped_extension_fits tenv _ Hfits_b Hscoped_b).
    fold level_a in Hextension_a. fold level_b in Hextension_b.
    injection Hmeet as <-.
    split.
    + intros r menv' Heq. injection Heq as <- <-.
      destruct (join_return_env_keeps join Hjoin choices Hchoice mh Hkeeps
                  Hsound menv base_a env_a base_b env_b Hfits Hbase_a
                  (fun x Hx => Hx) Hextends_a Hfits_a Hnames_a Hbase_b
                  (fun x Hx => Hx) Hextends_b Hfits_b Hnames_b result_env
                  menv1 menv2 Hjoined Hdefined_a Hdefined_b)
        as (Hextends & Hfits' & Hnames).
      split; [ exact Hextends | ]. split; [ exact Hfits' | ].
      split; [ | exact Hnames ].
      match goal with
      | |- fits _ (extract_value (combine_meet_return_values ?rx ?ry ?f) _ _)
        =>
        destruct (extract_combine rx ry f x1 x2) as [ -> | [ -> | -> ] ]
      end.
      * exact (proj1 disjunction_stable menv menv2 x1 Hextends Hx1).
      * exact (proj1 disjunction_stable menv menv2 x2 Hextends Hx2).
      * split; [ exact (Hkinds_a _ _ _ Hknows_a Hanswer_a) | ].
        split; [ exact (Hkinds_b _ _ _ Hknows_b Hanswer_b) | ].
        unfold extensions.
        destruct (Typing_env_level.is_empty level_a
                  && Typing_env_level.is_empty level_b);
          [ split; exact I | ].
        split.
        -- exact (extension_fits_kept _ _ _ Hknows_a Hextension_a).
        -- exact (extension_fits_kept _ _ _ Hknows_b Hextension_b).
    + intros statics valuation Hstatics Hsatisfied Hv1 Hv2.
      assert (Hbase_valued :=
        increment_scope_valued tenv valuation (proj1 Hfits)).
      destruct (Hsides statics valuation Hv1 Hv2) as [ [Hs1 Hs2] | [Hs1 Hs2] ].
      * destruct (Hlost_a statics valuation Hstatics Hsatisfied Hs1 Hs2)
          as (r & env_c & valuation_a & Heq & Hextended_a & Hsatisfied_a
              & Htrue_a & Hvalued_a).
        injection Heq as <- <-.
        destruct (join_return_env_sound join Hjoin choices Hchoice mh Hkeeps
                    Hsound menv base_a env_a base_b env_b Hfits Hbase_a
                    Hextends_a Hfits_a Hnames_a Hbase_b
                    (fun x Hx => Hx) Hextends_b Hfits_b Hnames_b result_env
                    menv1 menv2 Hjoined Hdefined_a Hdefined_b statics
                    valuation valuation_a (Meet_env.typing_env env_a)
                    (or_introl eq_refl) Hstatics Hsatisfied Hextended_a
                    Hsatisfied_a
                    Hvalued_a)
          as (valuation' & Hextended' & Hsatisfied' & Hvalued').
        assert (Hextended : extended_from tenv valuation valuation').
        { exact (extended_from_trans _ _ _ _ _
                   (scoped_keeps_names tenv _ Hscoped_a) Hextended_a
                   Hextended'). }
        eexists. exists menv2, valuation'. split; [ reflexivity | ].
        split; [ exact Hextended | ]. split; [ exact Hsatisfied' | ].
        split; [ | exact Hvalued' ].
        match goal with
        | |- satisfied _ _
               (extract_value (combine_meet_return_values ?rx ?ry ?f) _ _) =>
          destruct (extract_combine rx ry f x1 x2) as [ -> | [ -> | -> ] ]
        end.
        -- exact (proj2 disjunction_stable statics tenv valuation valuation'
                    x1 Hx1 Hextended Hv1).
        -- exact (proj2 disjunction_stable statics tenv valuation valuation'
                    x2 Hx2 Hextended Hv2).
        -- left. split.
           ++ exact (proj2 Hstable_a statics _ valuation_a valuation' _
                       Hanswer_a Hextended' Htrue_a).
           ++ unfold extensions.
              destruct (Typing_env_level.is_empty level_a
                        && Typing_env_level.is_empty level_b);
                [ exact I | ].
              apply (env_extension_satisfied_extended statics _ valuation_a
                       valuation' _ (proj1 (proj2 Hextension_a)) Hextended').
              exact (scoped_extension_satisfied tenv _ Hfits_a Hscoped_a
                       statics valuation_a Hsatisfied_a
                       (Hvalued_a _ Hbase_valued)).
      * destruct (Hlost_b statics valuation Hstatics Hsatisfied Hs1 Hs2)
          as (r & env_c & valuation_b & Heq & Hextended_b & Hsatisfied_b
              & Htrue_b & Hvalued_b).
        injection Heq as <- <-.
        rewrite Hbase_b in Hvalued_b.
        destruct (join_return_env_sound join Hjoin choices Hchoice mh Hkeeps
                    Hsound menv base_a env_a base_b env_b Hfits Hbase_a
                    Hextends_a Hfits_a Hnames_a Hbase_b
                    (fun x Hx => Hx) Hextends_b Hfits_b Hnames_b result_env
                    menv1 menv2 Hjoined Hdefined_a Hdefined_b statics
                    valuation valuation_b (Meet_env.typing_env env_b)
                    (or_intror eq_refl) Hstatics Hsatisfied Hextended_b
                    Hsatisfied_b
                    Hvalued_b)
          as (valuation' & Hextended' & Hsatisfied' & Hvalued').
        assert (Hextended : extended_from tenv valuation valuation').
        { exact (extended_from_trans _ _ _ _ _
                   (scoped_keeps_names tenv _ Hscoped_b) Hextended_b
                   Hextended'). }
        eexists. exists menv2, valuation'. split; [ reflexivity | ].
        split; [ exact Hextended | ]. split; [ exact Hsatisfied' | ].
        split; [ | exact Hvalued' ].
        match goal with
        | |- satisfied _ _
               (extract_value (combine_meet_return_values ?rx ?ry ?f) _ _) =>
          destruct (extract_combine rx ry f x1 x2) as [ -> | [ -> | -> ] ]
        end.
        -- exact (proj2 disjunction_stable statics tenv valuation valuation'
                    x1 Hx1 Hextended Hv1).
        -- exact (proj2 disjunction_stable statics tenv valuation valuation'
                    x2 Hx2 Hextended Hv2).
        -- right. split.
           ++ exact (proj2 Hstable_b statics _ valuation_b valuation' _
                       Hanswer_b Hextended' Htrue_b).
           ++ unfold extensions.
              destruct (Typing_env_level.is_empty level_a
                        && Typing_env_level.is_empty level_b);
                [ exact I | ].
              apply (env_extension_satisfied_extended statics _ valuation_b
                       valuation' _ (proj1 (proj2 Hextension_b)) Hextended').
              exact (scoped_extension_satisfied tenv _ Hfits_b Hscoped_b
                       statics valuation_b Hsatisfied_b
                       (Hvalued_b _ Hbase_valued)).
Qed.

End Disjunction.

Lemma meet_disjunction_ok_for :
  for_validity meet_disjunction_ok_statement.
Proof.
  intros A B fits_a satisfied_a fits_b satisfied_b join choices mh meet_a
    meet_b bottom_a bottom_b Hjoin Hchoice Hkeeps Hsound Hstable_a Hstable_b
    Hkinds_a Hkinds_b Hmeet_a Hmeet_b Hexclusive Hbottom_a Hbottom_b menv
    [[a1 b1] e1] [[a2 b2] e2] result Hfits Hx1 Hx2 Hmeet.
  exact (meet_disjunction_call fits_a satisfied_a fits_b satisfied_b
           Hstable_a Hstable_b join choices mh meet_a meet_b bottom_a bottom_b
           Hjoin Hchoice Hkeeps Hsound Hkinds_a Hkinds_b Hmeet_a Hmeet_b
           Hexclusive Hbottom_a Hbottom_b menv a1 a2 b1 b2 e1 e2 Hfits Hx1 Hx2
           result Hmeet).
Qed.

End Validity.

Theorem meet_disjunction_ok :
  meet_disjunction_ok_statement.
Proof.
  first
    [ exact (@meet_disjunction_ok_for standard_validity)
    | exact meet_disjunction_ok_for ].
Qed.
