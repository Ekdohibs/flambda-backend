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


(* Proofs of the statements of FlambdaJoinSpec.v. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs FlambdaJoinEnv
  FlambdaJoinTypes FlambdaJoinSpec.
Import ListNotations.

(* ======================================================================== *)
(* Small facts                                                              *)
(* ======================================================================== *)

(* A variable that may occur in terms is known. *)
Lemma in_scope_in_terms_knows (tenv : Typing_env.t) (x : variable) :
  in_scope_in_terms tenv x -> knows tenv x.
Proof.
  unfold in_scope_in_terms, knows, Typing_env.mem.
  destruct
    (Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv))
    as [[ty binding_time_and_mode] | ]; [ reflexivity | discriminate ].
Qed.

(* The environment that knows nothing is satisfied strictly by some
   valuation. *)
Lemma create_satisfied_strictly (statics : static_env) :
  ∃ valuation : env,
    typing_env_satisfied_strictly statics valuation Typing_env.create.
Proof.
  destruct (create_satisfied statics []) as [_ [valuation [Hsatisfied _]]].
  exists valuation. exact Hsatisfied.
Qed.

(* The one use of a list of one. *)
Lemma use_at_singleton {A : Type} (source : Typing_env.t) (scope : Scope.t)
    (id : A) (tenv use : Typing_env.t) (index : Index.t) :
  use_at {| source := source; cut_after := scope; uses := [(id, tenv)] |}
    index use ->
  use = tenv.
Proof.
  intros [id' Hnth]. cbn in Hnth. destruct index as [ | index]; cbn in Hnth.
  - injection Hnth as _ <-. reflexivity.
  - destruct index; discriminate.
Qed.

(* ======================================================================== *)
(* The contract is neither empty nor met by everything                      *)
(* ======================================================================== *)

Theorem join_to_source_sound : join_to_source_sound_statement.
Proof.
  intros typing_env_well_formed A avoid inputs target analysis
    [[Hsource _] [[Hno_bottom _] _]] _ Hjoin.
  injection Hjoin as <- <-.
  split; [ exact Hsource | ]. split; [ exact Hno_bottom | ].
  split; [ | split; [ | split ] ].
  - split; [ | repeat split ].
    intros n ty binding_time_and_mode Hlookup. exists ty. exact Hlookup.
  - intros x Hknown Hnot_known. contradiction.
  - split; [ | split ].
    + intros index id Hlookup. discriminate.
    + intros x simples kind index simple Hlookup. discriminate.
    + intros x simples kind index simple Hlookup. discriminate.
  - intros index use Huse statics valuation _ Hsatisfied.
    exists valuation. split; [ exact Hsatisfied | ].
    split; [ intros x _; reflexivity | ].
    split; [ intros x _; reflexivity | ].
    split; intros x simples kind simple v Hlookup; discriminate.
Qed.

Theorem sound_join_where_values_are_read_of_sound
  : sound_join_where_values_are_read_of_sound_statement.
Proof.
  intros typing_env_well_formed A join Hsound avoid inputs target analysis
    Hwell_formed Hkinded Hjoin.
  destruct (Hsound avoid inputs target analysis Hwell_formed Hkinded Hjoin)
    as [Htarget [Hno_bottom [Hkeeps [Hintroduces [Hanalysis Hadmits]]]]].
  split; [ exact Htarget | ]. split; [ exact Hno_bottom | ].
  split; [ exact Hkeeps | ].
  split; [ exact Hintroduces | ]. split; [ exact Hanalysis | ].
  intros index use Huse statics valuation Hsatisfied Hsource _.
  exact (Hadmits index use Huse statics valuation Hsatisfied Hsource).
Qed.

Theorem equations_have_values_of_values_are_read
  : equations_have_values_of_values_are_read_statement.
Proof.
  intros typing_env_well_formed A join Hsound avoid inputs target analysis
    Hwell_formed Hkinded Hjoin.
  destruct (Hsound avoid inputs target analysis Hwell_formed Hkinded Hjoin)
    as [Htarget [Hno_bottom [Hkeeps [Hintroduces [Hanalysis Hadmits]]]]].
  split; [ exact Htarget | ]. split; [ exact Hno_bottom | ].
  split; [ exact Hkeeps | ].
  split; [ exact Hintroduces | ]. split; [ exact Hanalysis | ].
  intros index use Huse statics valuation Hsatisfied Hsource [Hread Hadded].
  destruct (Hadmits index use Huse statics valuation Hsatisfied Hsource Hread)
    as [valuation'
          [Htarget' [Hagrees [Hagrees_source [Hdefinitions Hnormal]]]]].
  exists valuation'.
  split; [ exact Htarget' | ]. split; [ exact Hagrees | ].
  split; [ exact Hagrees_source | ]. split; [ exact Hdefinitions | ].
  split; [ exact Hnormal | ].
  intros x ty Hequation.
  destruct (Hadded x ty Hequation)
    as [[Hknown [v Hv]]
       | [simples [kind [simple [v [Hdefinition [Hfind Hv]]]]]]].
  - exists v. destruct Hknown as [Hknown | [use' [Huse' Hknown]]].
    + rewrite (Hagrees_source x Hknown). exact Hv.
    + assert (Hsame : use' = use).
      { destruct Huse as [id Hid]. destruct Huse' as [id' Hid'].
        rewrite Hid in Hid'. injection Hid' as _ Hsame. symmetry.
        exact Hsame. }
      subst use'. rewrite (Hagrees x Hknown). exact Hv.
  - exists v. exact (Hdefinitions x simples kind simple v Hdefinition Hfind Hv).
Qed.

Theorem join_to_bottom_unsound : join_to_bottom_unsound_statement.
Proof.
  intros Hsound.
  pose (inputs :=
    {| source := Typing_env.create;
       cut_after := Typing_env.current_scope Typing_env.create;
       uses := [(tt, Typing_env.create)] |}).
  assert (Hwell_formed : join_inputs_well_formed (fun _ => True) [] inputs).
  { split; [ split; [ exact I | intros; exact I ] | ].
    split.
    { split.
      - intros _ n ty binding_time_and_mode Hlookup. discriminate.
      - intros index use Huse _ n ty binding_time_and_mode Hlookup.
        apply use_at_singleton in Huse. subst use. discriminate. }
    split; [ | split ].
    - intros index use Huse _.
      apply use_at_singleton in Huse. subst use.
      exists Typing_env_level.empty. reflexivity.
    - intros index use s Huse Hdefined.
      apply use_at_singleton in Huse. subst use. exact Hdefined.
    - split.
      + intros x Hknown. discriminate.
      + intros index use x Huse Hknown.
        apply use_at_singleton in Huse. subst use. discriminate. }
  assert (Hkinded : uses_kinded inputs).
  { intros index use Huse. apply use_at_singleton in Huse. subst use.
    exact create_kinded. }
  destruct (Hsound [] inputs _ _ Hwell_formed Hkinded eq_refl)
    as [_ [_ [_ [_ [_ Hadmits]]]]].
  assert (Huse : use_at inputs 0 Typing_env.create).
  { exists tt. reflexivity. }
  destruct (create_satisfied_strictly (Static_env [] []))
    as [valuation Hsatisfied].
  destruct
    (Hadmits 0 Typing_env.create Huse _ valuation Hsatisfied Hsatisfied)
    as [valuation' [Htarget _]].
  exact
    (bottom_not_satisfied_but (fun _ => False) _ valuation'
       (Typing_env.make_bottom (source inputs)) eq_refl Htarget).
Qed.

(* ======================================================================== *)
(* In the machine                                                           *)
(* ======================================================================== *)

(* A [Simple] that may occur in terms at a use has a value in the
   machine, which a valuation that agrees with it has too. *)
Lemma eval_simple_in_terms (use : Typing_env.t) (valuation vars : env)
    (simple : simple) :
  bound_in_terms use vars ->
  agrees_in_terms use valuation vars ->
  Typing_env.mem_simple (Some Name_mode.normal) use simple = true ->
  ∃ v : value,
    eval_simple vars simple = Some v ∧ eval_simple valuation simple = Some v.
Proof.
  intros Hbound Hterms Hnormal.
  destruct simple as [y | s | c]; cbn in *.
  - destruct (Hbound y Hnormal) as [v Hv]. exists v.
    split; [ exact Hv | ]. rewrite (Hterms y Hnormal). exact Hv.
  - eexists. split; reflexivity.
  - eexists. split; reflexivity.
Qed.

Theorem join_admits_use_in_machine : join_admits_use_in_machine_statement.
Proof.
  intros source use target definitions definitions_at_normal_mode index
    statics vars Hadmits Hscope
    [Hbound [valuation [Huse [Hsource Hterms]]]].
  destruct (Hadmits statics valuation Huse Hsource)
    as [valuation' [Htarget [Hagrees [_ [_ Hdefinitions]]]]].
  exists valuation'. split; [ exact Htarget | ].
  split; [ | split ].
  - intros x Hx. specialize (Hscope x Hx).
    rewrite (Hagrees x (in_scope_in_terms_knows _ _ Hscope)).
    exact (Hterms x Hscope).
  - intros x Hx. apply Hbound. apply Hscope. exact Hx.
  - intros x simples kind simple Hlookup Hfind Hnormal.
    destruct (eval_simple_in_terms use valuation vars simple Hbound Hterms
                Hnormal)
      as [v [Hvars Hvaluation]].
    rewrite Hvars.
    exact (Hdefinitions x simples kind simple v Hlookup Hfind Hvaluation).
Qed.

Theorem join_admits_use_env_satisfies
  : join_admits_use_env_satisfies_statement.
Proof.
  intros source use target definitions definitions_at_normal_mode index
    statics vars Hadmits Hscope Hsatisfied.
  destruct
    (join_admits_use_in_machine source use target definitions
       definitions_at_normal_mode index statics vars Hadmits Hscope
       Hsatisfied)
    as [valuation' [Htarget [Hterms [Hbound _]]]].
  split; [ exact Hbound | ]. exists valuation'. split; assumption.
Qed.

(* Whether a variable may occur in terms depends on its binding time and
   name mode and on the first binding time in scope. *)
Lemma in_scope_in_terms_kept (source target : Typing_env.t) (x : variable) :
  join_keeps_source source target ->
  knows source x ->
  in_scope_in_terms target x ->
  in_scope_in_terms source x.
Proof.
  intros [Hnames [_ [_ [_ Hmin]]]].
  unfold knows, in_scope_in_terms, Typing_env.mem.
  destruct
    (Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types source))
    as [[ty binding_time_and_mode] | ] eqn:Hsource; [ | discriminate ].
  destruct (Hnames _ _ _ Hsource) as [ty' Htarget].
  rewrite Htarget, Hmin. intros _ Hscope. exact Hscope.
Qed.

Theorem join_guarantees_in_machine : join_guarantees_in_machine_statement.
Proof.
  intros typing_env_well_formed A avoid inputs target analysis index use
    statics vars [_ [_ [Hkeeps [Hintroduces [_ Hadmits]]]]] Hterms Huse
    Hsatisfied.
  apply
    (join_admits_use_env_satisfies (source inputs) use target
       (Analysis.definitions_in_joined_envs analysis)
       (Analysis.canonical_definitions_at_normal_mode analysis) index
       statics vars (Hadmits index use Huse)); [ | exact Hsatisfied ].
  intros x Hscope. apply (Hterms index use x Huse).
  destruct (Typing_env.mem None (source inputs) (Name_var x)) eqn:Hknown.
  - exact (in_scope_in_terms_kept _ _ x Hkeeps Hknown Hscope).
  - destruct (Hintroduces x (in_scope_in_terms_knows _ _ Hscope))
      as [Hnot_in_scope _].
    + unfold knows. rewrite Hknown. discriminate.
    + contradiction.
Qed.

Theorem known_value_at_use : known_value_at_use_statement.
Proof.
  intros definitions index valuation valuation' x simples kind simple c
    Hdefinitions Hlookup Hfind Hvalue.
  apply (Hdefinitions x simples kind simple _ Hlookup Hfind).
  destruct simple as [y | s | c']; cbn in Hvalue; try discriminate.
  injection Hvalue as <-. reflexivity.
Qed.

(* A binding of the bindings that count is what the map has for its
   key. *)
Lemma in_compact_lookup {A : Type} (bindings : Keyed.finite_map Index.t A)
    (index : Index.t) (a : A) :
  In (index, a) (Keyed.compact Index.equal bindings) ->
  Keyed.lookup Index.equal index bindings = Some a.
Proof.
  unfold Keyed.compact, Keyed.tabulate. rewrite in_flat_map.
  intros [key [_ Hin]].
  destruct (Keyed.lookup Index.equal key bindings) as [a' | ] eqn:Hlookup;
    [ | destruct Hin ].
  destruct Hin as [Heq | []]. injection Heq as <- <-. exact Hlookup.
Qed.

Lemma values_of_bindings_found {A : Type}
    (external_ids : Keyed.finite_map Index.t A)
    (bindings : list (Index.t * simple))
    (values : list (A * Or_unknown.t const)) (id : A)
    (value : Or_unknown.t const) :
  Analysis.values_of_bindings external_ids bindings = Some values ->
  In (id, value) values ->
  ∃ (index : Index.t) (simple : simple),
    Keyed.lookup Index.equal index external_ids = Some id
    ∧ In (index, simple) bindings
    ∧ value = Analysis.value_at_use simple.
Proof.
  revert values. induction bindings as [ | [index simple] bindings IH];
    intros values Hvalues Hin; cbn in Hvalues.
  - injection Hvalues as <-. destruct Hin.
  - destruct (Keyed.lookup Index.equal index external_ids)
      as [external_id | ] eqn:Hid; [ | discriminate ].
    destruct (Analysis.values_of_bindings external_ids bindings)
      as [values' | ]; [ | discriminate ].
    injection Hvalues as <-. destruct Hin as [Heq | Hin].
    + injection Heq as <- <-. exists index, simple.
      split; [ exact Hid | ]. split; [ left; reflexivity | reflexivity ].
    + destruct (IH values' eq_refl Hin) as [index' [simple' [H1 [H2 H3]]]].
      exists index', simple'. split; [ exact H1 | ].
      split; [ right; exact H2 | exact H3 ].
Qed.

Theorem values_at_uses_found : values_at_uses_found_statement.
Proof.
  intros A external_ids simples values id value Hvalues Hin.
  destruct (values_of_bindings_found _ _ _ _ _ Hvalues Hin)
    as [index [simple [Hid [Hbinding Hvalue]]]].
  exists index, simple. split; [ exact Hid | ].
  split; [ exact (in_compact_lookup _ _ _ Hbinding) | exact Hvalue ].
Qed.

Lemma definitions_of_bindings_found {A : Type}
    (external_ids : Keyed.finite_map Index.t A)
    (bindings : list (Index.t * simple)) (definitions : list (A * simple))
    (id : A) (simple : simple) :
  Analysis.definitions_of_bindings external_ids bindings = Some definitions ->
  In (id, simple) definitions ->
  ∃ index : Index.t,
    Keyed.lookup Index.equal index external_ids = Some id
    ∧ In (index, simple) bindings.
Proof.
  revert definitions. induction bindings as [ | [index simple'] bindings IH];
    intros definitions Hdefinitions Hin; cbn in Hdefinitions.
  - injection Hdefinitions as <-. destruct Hin.
  - destruct (Keyed.lookup Index.equal index external_ids)
      as [external_id | ] eqn:Hid; [ | discriminate ].
    destruct (Analysis.definitions_of_bindings external_ids bindings)
      as [definitions' | ]; [ | discriminate ].
    injection Hdefinitions as <-. destruct Hin as [Heq | Hin].
    + injection Heq as <- <-. exists index.
      split; [ exact Hid | left; reflexivity ].
    + destruct (IH definitions' eq_refl Hin) as [index' [H1 H2]].
      exists index'. split; [ exact H1 | right; exact H2 ].
Qed.

Theorem definitions_at_uses_found : definitions_at_uses_found_statement.
Proof.
  intros A external_ids simples definitions id simple Hdefinitions Hin.
  destruct (definitions_of_bindings_found _ _ _ _ _ Hdefinitions Hin)
    as [index [Hid Hbinding]].
  exists index. split; [ exact Hid | ].
  exact (in_compact_lookup _ _ _ Hbinding).
Qed.

(* ======================================================================== *)
(* The n-way join of closures entries                                       *)
(* ======================================================================== *)

(* The instance is that of the examples of the meaning of types: the
   type that a projection of a value slot gives a closure, joined with
   one that names a code other than the closure's. *)
Theorem n_way_join_function_types_before_fix_unsound
  : n_way_join_function_types_before_fix_unsound_statement.
Proof.
  exists Examples.statics, Examples.vars, (Value_symbol Examples.a_closure),
    unnamed_function_slot,
    (Row_like_index
       (Row_like_index_domain.At_least
          (Set_of_closures_contents [] [Examples.a_value_slot]))
       tt),
    [], [(Examples.a_value_slot, unknown Value)], [],
    [(unnamed_function_slot,
      Or_unknown.Known (Function_type Examples.other_code))].
  split.
  - exact Examples.nway_join_input_has.
  - exact Examples.nway_join_has_not.
Qed.

(* What a join of function types that answers a code says of each of
   the function types joined. *)
Lemma known_function_types_in
    (func_types : list (Index.t * Or_unknown.t function_type))
    (known : list function_type) (index : Index.t)
    (func_type : Or_unknown.t function_type) :
  known_function_types func_types = Some known ->
  In (index, func_type) func_types ->
  ∃ func_type' : function_type,
    func_type = Or_unknown.Known func_type' ∧ In func_type' known.
Proof.
  revert known. induction func_types as [ | [index' func_type'] func_types IH];
    intros known Hknown Hin; [ destruct Hin | ].
  cbn in Hknown. destruct func_type' as [func_type' | ]; [ | discriminate ].
  destruct (known_function_types func_types) as [known' | ];
    [ | discriminate ].
  injection Hknown as <-. destruct Hin as [Heq | Hin].
  - injection Heq as _ <-. exists func_type'. split; [ reflexivity | ].
    left. reflexivity.
  - destruct (IH known' eq_refl Hin) as [func_type'' [Heq Hin']].
    exists func_type''. split; [ exact Heq | ]. right. exact Hin'.
Qed.

Lemma n_way_join_function_type_known
    (func_types : list (Index.t * Or_unknown.t function_type))
    (joined : function_type) (index : Index.t)
    (func_type : Or_unknown.t function_type) :
  n_way_join_function_type func_types = Or_unknown.Known joined ->
  In (index, func_type) func_types ->
  ∃ func_type' : function_type,
    func_type = Or_unknown.Known func_type'
    ∧ function_type_code_id func_type' = function_type_code_id joined.
Proof.
  unfold n_way_join_function_type. intros Hjoined Hin.
  destruct (known_function_types func_types) as [known | ] eqn:Hknown;
    [ | discriminate ].
  destruct known as [ | first known]; [ discriminate | ].
  destruct
    (forallb
       (fun func_type' =>
          Pos.eqb (function_type_code_id first)
            (function_type_code_id func_type'))
       known) eqn:Hall; [ | discriminate ].
  injection Hjoined as <-.
  destruct (known_function_types_in _ _ _ _ Hknown Hin)
    as [func_type' [Heq [Hfirst | Hrest]]].
  - exists func_type'. split; [ exact Heq | ]. subst func_type'. reflexivity.
  - exists func_type'. split; [ exact Heq | ].
    rewrite forallb_forall in Hall. specialize (Hall func_type' Hrest).
    apply Pos.eqb_eq in Hall. symmetry. exact Hall.
Qed.

(* One step of the gathering. *)
Definition gather_step
    (function_types : finite_map (list (Index.t * Or_unknown.t function_type)))
    (entry : Index.t * finite_map (Or_unknown.t function_type))
    : finite_map (list (Index.t * Or_unknown.t function_type)) :=
  merge
    (fun vals_opt val2_opt =>
       match vals_opt, val2_opt with
       | None, _ | _, None => None
       | Some vals, Some val2 => Some ((fst entry, val2) :: vals)
       end)
    function_types (snd entry).

Lemma lookup_gather_step
    (function_types : finite_map (list (Index.t * Or_unknown.t function_type)))
    (entry : Index.t * finite_map (Or_unknown.t function_type))
    (g : function_slot)
    (contributors : list (Index.t * Or_unknown.t function_type)) :
  lookup g (gather_step function_types entry) = Some contributors ->
  ∃ (vals : list (Index.t * Or_unknown.t function_type))
    (val2 : Or_unknown.t function_type),
    lookup g function_types = Some vals
    ∧ lookup g (snd entry) = Some val2
    ∧ contributors = (fst entry, val2) :: vals.
Proof.
  unfold gather_step. rewrite lookup_merge; [ | reflexivity ].
  destruct (lookup g function_types) as [vals | ]; [ | discriminate ].
  destruct (lookup g (snd entry)) as [val2 | ]; [ | discriminate ].
  intros Heq. injection Heq as <-. exists vals, val2. repeat split.
Qed.

(* What is gathered for a slot holds a function type of every entry gone
   through, and what was gathered before. *)
Lemma lookup_fold_gather
    (entries : list (Index.t * finite_map (Or_unknown.t function_type)))
    (function_types : finite_map (list (Index.t * Or_unknown.t function_type)))
    (g : function_slot)
    (contributors : list (Index.t * Or_unknown.t function_type)) :
  lookup g (List.fold_left gather_step entries function_types)
  = Some contributors ->
  (∃ vals : list (Index.t * Or_unknown.t function_type),
     lookup g function_types = Some vals
     ∧ ∀ contributor, In contributor vals -> In contributor contributors)
  ∧ ∀ (index : Index.t)
      (function_types' : finite_map (Or_unknown.t function_type)),
      In (index, function_types') entries ->
      ∃ func_type : Or_unknown.t function_type,
        lookup g function_types' = Some func_type
        ∧ In (index, func_type) contributors.
Proof.
  revert function_types.
  induction entries as [ | entry entries IH]; intros function_types Hlookup.
  - split; [ | intros index function_types' [] ].
    exists contributors. split; [ exact Hlookup | ]. intros; assumption.
  - cbn in Hlookup.
    destruct (IH _ Hlookup) as [[vals1 [Hstep Hincluded]] Hentries].
    destruct (lookup_gather_step _ _ _ _ Hstep)
      as [vals [val2 [Hvals [Hval2 Heq]]]].
    subst vals1. split.
    + exists vals. split; [ exact Hvals | ].
      intros contributor Hin. apply Hincluded. right. exact Hin.
    + intros index function_types' [Heq | Hin].
      * subst entry. exists val2. split; [ exact Hval2 | ].
        apply Hincluded. left. reflexivity.
      * exact (Hentries index function_types' Hin).
Qed.

Lemma lookup_gather_function_types
    (entries : list (Index.t * finite_map (Or_unknown.t function_type)))
    (g : function_slot)
    (contributors : list (Index.t * Or_unknown.t function_type))
    (index : Index.t)
    (function_types : finite_map (Or_unknown.t function_type)) :
  lookup g (gather_function_types entries) = Some contributors ->
  In (index, function_types) entries ->
  ∃ func_type : Or_unknown.t function_type,
    lookup g function_types = Some func_type
    ∧ In (index, func_type) contributors.
Proof.
  destruct entries as [ | [id1 function_types1] entries];
    intros Hlookup Hin; [ destruct Hin | ].
  cbn in Hlookup.
  change
    (lookup g
       (List.fold_left gather_step entries
          (map_values (fun func_type => [(id1, func_type)]) function_types1))
     = Some contributors) in Hlookup.
  destruct (lookup_fold_gather _ _ _ _ Hlookup)
    as [[vals [Hvals Hincluded]] Hentries].
  destruct Hin as [Heq | Hin]; [ | exact (Hentries _ _ Hin) ].
  injection Heq as <- <-.
  rewrite lookup_map_values in Hvals.
  destruct (lookup g function_types1) as [func_type | ]; [ | discriminate ].
  injection Hvals as <-. exists func_type. split; [ reflexivity | ].
  apply Hincluded. left. reflexivity.
Qed.

Theorem n_way_join_function_types_sound
  : n_way_join_function_types_sound_statement.
Proof.
  intros statics vars v slot functions value_slots entries index
    function_types closure_types value_slot_types Hin.
  rewrite !closures_entry_lookup_iff.
  intros [Hfunctions Hrest]. split; [ | exact Hrest ].
  intros g joined code Hlookup Hcode.
  unfold n_way_join_function_types in Hlookup.
  rewrite lookup_map_values in Hlookup.
  destruct (lookup g (gather_function_types entries))
    as [contributors | ] eqn:Hgathered; [ | discriminate ].
  injection Hlookup as Hjoined.
  destruct
    (lookup_gather_function_types _ _ _ _ _ Hgathered Hin)
    as [func_type [Hfunc_type Hcontributor]].
  destruct (n_way_join_function_type_known _ _ _ _ Hjoined Hcontributor)
    as [func_type' [Heq Hsame_code]].
  subst func_type. rewrite <- Hsame_code.
  exact (Hfunctions g func_type' code Hfunc_type Hcode).
Qed.
