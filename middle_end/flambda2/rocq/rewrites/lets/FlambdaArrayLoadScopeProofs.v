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


(* Proof of the statement of FlambdaArrayLoadScopeSpec.v. Nothing here
   needs review. It follows the proof for the load from a known block
   in FlambdaKeepsScopeProofs.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaNormaliseSpec
  FlambdaPrimitiveModel FlambdaPrimitiveSpec FlambdaBlockLoadSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaArrayLoadSpec FlambdaArrayLoadScopeSpec.
Import ListNotations.

(* The variables that the elements of a known array are known to hold
   are among [names]. *)
Definition array_within (names : list variable) (array : known_array)
    : Prop :=
  ∀ s : simple, In (Some s) array -> incl (free_variables_of_simple s) names.

Definition arrays_within (names : list variable) (known : known_arrays)
    : Prop :=
  (∀ entry : variable * known_array,
    In entry (known_array_vars known) -> array_within names (snd entry))
  ∧ (∀ entry : symbol * known_array,
      In entry (known_array_symbols known) ->
      array_within names (snd entry)).

Lemma arrays_within_weaken (names names' : list variable)
    (known : known_arrays) :
  incl names names' -> arrays_within names known -> arrays_within names' known.
Proof.
  intros Hincl [Hvars Hsymbols].
  split; intros entry Hentry s Hs;
    (eapply incl_tran; [ | exact Hincl ]);
    [ exact (Hvars entry Hentry s Hs) | exact (Hsymbols entry Hentry s Hs) ].
Qed.

Lemma forget_in_array_within (names : list variable) (x : variable)
    (array : known_array) :
  array_within names array -> array_within names (forget_in_array x array).
Proof.
  intros Harray s Hs. unfold forget_in_array in Hs.
  apply in_map_iff in Hs. destruct Hs as (field & Hfield & Hin).
  destruct field as [s' | ]; cbn [forget_field] in Hfield; [ | discriminate ].
  destruct (var_in_simple x s'); [ discriminate | ].
  injection Hfield as ->. exact (Harray s Hin).
Qed.

Lemma forget_array_var_within (names : list variable) (x : variable)
    (known : known_arrays) :
  arrays_within names known ->
  arrays_within names (forget_array_var x known).
Proof.
  intros [Hvars Hsymbols]. unfold forget_array_var.
  split; cbn [known_array_vars known_array_symbols]; intros entry Hentry;
    apply in_map_iff in Hentry; destruct Hentry as (entry' & <- & Hin);
    cbn [snd]; apply forget_in_array_within.
  - apply filter_In in Hin. exact (Hvars entry' (proj1 Hin)).
  - exact (Hsymbols entry' Hin).
Qed.

Lemma forget_array_vars_within (names : list variable)
    (xs : list variable) :
  ∀ known : known_arrays,
    arrays_within names known ->
    arrays_within names (forget_array_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known Hknown;
    cbn [forget_array_vars];
    [ exact Hknown | apply IH, forget_array_var_within, Hknown ].
Qed.

(* The elements of an array that is made are among the arguments of
   what makes it. *)
Lemma elements_within (names : list variable) (elements : list simple) :
  incl (free_variables_of_simples elements) names ->
  array_within names (map Some elements).
Proof.
  intros Hincl s Hs. apply in_map_iff in Hs.
  destruct Hs as (s' & Heq & Hin). injection Heq as ->.
  intros y Hy. apply Hincl. unfold free_variables_of_simples.
  apply in_flat_map. exists s. split; assumption.
Qed.

Lemma arrays_under_let_within (names : list variable)
    (known : known_arrays) (x : variable) (d : named) :
  arrays_within names known ->
  arrays_within (x :: names ++ free_variables_of_named d)
    (arrays_under_let known x d).
Proof.
  intros Hknown. unfold arrays_under_let. cbv zeta.
  assert (Hforgot :
    arrays_within (x :: names ++ free_variables_of_named d)
      (forget_array_var x known)).
  { apply forget_array_var_within.
    eapply arrays_within_weaken; [ | exact Hknown ].
    apply incl_tl, incl_appl, incl_refl. }
  destruct (array_made_by d) as [array | ] eqn:Hmade; [ | exact Hforgot ].
  destruct Hforgot as [Hvars Hsymbols].
  split; cbn [known_array_vars known_array_symbols]; [ | exact Hsymbols ].
  intros entry [<- | Hentry]; [ | exact (Hvars entry Hentry) ].
  cbn [snd]. apply forget_in_array_within.
  unfold array_made_by in Hmade.
  repeat
    match type of Hmade with
    | match ?t with _ => _ end = Some _ => destruct t; try discriminate Hmade
    end.
  injection Hmade as <-. apply elements_within.
  cbn [free_variables_of_named]. apply incl_tl, incl_appr, incl_refl.
Qed.

Lemma arrays_under_statics_within (names : list variable)
    (known : known_arrays) (bindings : list static_binding) :
  arrays_within names known ->
  arrays_within (names ++ flat_map free_variables_of_binding bindings)
    (arrays_under_statics known bindings).
Proof.
  intros Hknown.
  destruct
    (arrays_within_weaken names
       (names ++ flat_map free_variables_of_binding bindings) known
       (incl_appl _ (incl_refl _)) Hknown)
    as [Hvars Hsymbols].
  unfold arrays_under_statics.
  split; cbn [known_array_vars known_array_symbols]; [ exact Hvars | ].
  intros entry Hentry. apply in_app_or in Hentry.
  destruct Hentry as [Hentry | Hentry]; [ | exact (Hsymbols entry Hentry) ].
  apply in_flat_map in Hentry. destruct Hentry as (binding & Hbinding & Hentry).
  unfold static_array_made in Hentry.
  destruct binding as [ | | sym const]; try solve [ destruct Hentry ].
  destruct const as [ | | | | | elements]; try solve [ destruct Hentry ].
  destruct Hentry as [<- | []]. cbn [snd]. apply elements_within.
  intros y Hy. apply in_or_app. right.
  apply in_flat_map.
  exists (Bind_symbol sym (Static_const_immutable_value_array elements)).
  split; [ exact Hbinding | ].
  cbn [free_variables_of_binding free_variables_of_static_const]. exact Hy.
Qed.

(* What the pass puts in place of a primitive mentions variables of
   [names] only. *)
Lemma of_known_array_free (names : list variable) (known : known_arrays)
    (p : prim) (args : list simple) (result : simple) :
  arrays_within names known ->
  of_known_array known p args = Some result ->
  incl (free_variables_of_simple result) names.
Proof.
  intros [Hvars Hsymbols] Hresult.
  assert (Hknown :
    ∀ (a : simple) (array : known_array),
      known_array_of known a = Some array -> array_within names array).
  { intros a array Hlookup.
    destruct a as [x | sym | c]; cbn [known_array_of] in Hlookup;
      [ | | discriminate ].
    - exact (Hvars (x, array) (lookup_in _ _ _ Hlookup)).
    - exact (Hsymbols (sym, array) (lookup_in _ _ _ Hlookup)). }
  unfold of_known_array in Hresult.
  destruct p as [u | b | vp | hp | rp]; try discriminate Hresult.
  - destruct u; try discriminate Hresult.
    destruct args as [ | a [ | a' args]]; try discriminate Hresult.
    destruct (known_array_of known a) as [array | ];
      [ | discriminate Hresult ].
    unfold length_of_known in Hresult.
    destruct
      (in_range (int_width Standard_tagged_immediate)
         (Z.of_nat (length array)));
      [ | discriminate Hresult ].
    injection Hresult as <-. intros y [].
  - destruct b; try discriminate Hresult.
    destruct args as [ | a [ | index [ | arg args]]];
      try discriminate Hresult.
    destruct (known_array_of known a) as [array | ] eqn:Harray;
      [ | discriminate Hresult ].
    unfold element_loaded in Hresult.
    destruct index as [x | sym | c]; try discriminate Hresult.
    destruct c; try discriminate Hresult.
    destruct (0 <=? i)%Z; [ | discriminate Hresult ].
    destruct (nth_error array (Z.to_nat i)) as [[contents | ] | ] eqn:Hnth;
      try discriminate Hresult.
    injection Hresult as ->.
    exact (Hknown a array Harray result (nth_error_In _ _ Hnth)).
Qed.

Lemma load_known_array_named_free (names : list variable)
    (known : known_arrays) (d : named) :
  arrays_within names known ->
  incl (free_variables_of_named (load_known_array_named known d))
    (free_variables_of_named d ++ names).
Proof.
  intros Hknown. unfold load_known_array_named.
  destruct d as [ | p args | | | | | ]; try (apply incl_appl, incl_refl).
  destruct (of_known_array known p args) as [result | ] eqn:Hresult;
    [ | apply incl_appl, incl_refl ].
  cbn [free_variables_of_named]. apply incl_appr.
  exact (of_known_array_free names known p args result Hknown Hresult).
Qed.

Fixpoint load_arrays_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : known_arrays),
    arrays_within names known ->
    scoped_under names e (load_arrays_knowing known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown;
    cbn [load_arrays_knowing load_arrays_in_handler];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply load_known_array_named_free, Hknown | ].
    apply load_arrays_knowing_scoped, arrays_under_let_within, Hknown.
  - apply scoped_under_let_static.
    apply load_arrays_knowing_scoped, arrays_under_statics_within, Hknown.
  - apply scoped_under_let_cont; apply load_arrays_knowing_scoped;
      [ apply forget_array_vars_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply load_arrays_knowing_scoped, Hknown.
Qed.

Lemma load_known_arrays_scoped (e : expr) : scoped e (load_known_arrays e).
Proof.
  apply scoped_under_none, load_arrays_knowing_scoped.
  split; intros entry [].
Qed.

Theorem load_known_arrays_keeps_scope :
  load_known_arrays_keeps_scope_statement.
Proof. apply keeps_scope_intro, load_known_arrays_scoped. Qed.
